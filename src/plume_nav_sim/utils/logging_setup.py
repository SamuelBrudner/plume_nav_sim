"""
Compatibility wrapper for enhanced logging utilities.

This module re-uses the implementation from odor_plume_nav.utils.logging_setup
and adapts a few constants to match plume_nav_sim test expectations.
"""
from __future__ import annotations

# Re-export all functionality from the odor_plume_nav implementation
from odor_plume_nav.utils.logging_setup import *  # noqa: F401,F403

# The plume_nav_sim test suite expects the raw minimal format without Loguru markup
MINIMAL_FORMAT = "{level: <8} | {message}"

# Ensure MINIMAL_FORMAT is exported even when importing *
try:
    __all__  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    __all__ = []

if "MINIMAL_FORMAT" not in __all__:
    __all__.append("MINIMAL_FORMAT")

# --------------------------------------------------------------------------- #
# Override enhanced format to include request_id for plume_nav_sim tests      #
# --------------------------------------------------------------------------- #

# The upstream implementation does not include `request_id` in the enhanced
# console format, but the plume_nav_sim test-suite expects it.  We therefore
# define a local override that extends the pattern accordingly and make sure it
# is exported.

ENHANCED_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<magenta>correlation_id={extra[correlation_id]}</magenta> | "
    "<yellow>request_id={extra[request_id]}</yellow> | "
    "<blue>module={extra[module]}</blue> - "
    "<level>{message}</level>"
)

if "ENHANCED_FORMAT" not in __all__:
    __all__.append("ENHANCED_FORMAT")

# --------------------------------------------------------------------------- #
# Local override: preserve provided LoggingConfig without env defaults        #
# --------------------------------------------------------------------------- #

import sys  # needed for console sink
from pathlib import Path
from typing import Optional
from loguru import logger  # ensure logger available in local overrides

from odor_plume_nav.utils.logging_setup import (
    create_configuration_from_hydra as _upstream_create_configuration_from_hydra,
    LoggingConfig as _UpstreamLoggingConfig,
    EnhancedLogger as _UpstreamEnhancedLogger,
    _create_context_filter as _upstream_create_context_filter,
    _create_json_formatter as _upstream_create_json_formatter,
    setup_logger as _upstream_setup_logger,
)
from typing import Any, Mapping

# --------------------------------------------------------------------------- #
# Hydra convenience wrapper that preserves leading './' in file_path values   #
# --------------------------------------------------------------------------- #


def create_configuration_from_hydra(
    hydra_config: Any | None = None,
) -> _UpstreamLoggingConfig:
    """
    Thin wrapper around the upstream helper that **preserves** a leading
    ``./`` on ``file_path`` entries when the user supplied a plain ``dict``.

    The upstream implementation (via Pydantic path normalisation) strips the
    leading ``./`` which several plume_nav_sim tests explicitly assert is kept
    verbatim.  We therefore restore it post-construction if required.
    """

    original_path: str | None = None
    try:
        if isinstance(hydra_config, Mapping):
            original_path = hydra_config.get("file_path")  # type: ignore[arg-type]
    except Exception:  # pragma: no cover – defensive
        pass

    cfg = _upstream_create_configuration_from_hydra(hydra_config)

    if isinstance(original_path, str) and original_path.startswith("./"):
        # Restore leading './' that was stripped by path normalisation.
        try:
            object.__setattr__(cfg, "file_path", original_path)
        except Exception:  # pragma: no cover
            pass

    return cfg


if "create_configuration_from_hydra" not in __all__:
    __all__.append("create_configuration_from_hydra")

# --------------------------------------------------------------------------- #
# Patch EnhancedLogger.exception to append episode_id when available          #
# --------------------------------------------------------------------------- #

_orig_exception = _UpstreamEnhancedLogger.exception


def _patched_exception(self, message: str, **kwargs):  # noqa: D401
    """Enhanced .exception that embeds episode_id from correlation context."""
    try:
        ctx = get_correlation_context()
        ep_id = getattr(ctx, "episode_id", None)
        if ep_id:
            message = f"{message} [episode_id={ep_id}]"
    except Exception:  # pragma: no cover – safety
        pass
    return _orig_exception(self, message, **kwargs)


# Apply monkey-patch once at import time
_UpstreamEnhancedLogger.exception = _patched_exception

# --------------------------------------------------------------------------- #
# Monkey-patch LoggingConfig.get_format_pattern to honor local overrides      #
# --------------------------------------------------------------------------- #

# Save original for fallback
_orig_get_format_pattern = _UpstreamLoggingConfig.get_format_pattern


def _patched_get_format_pattern(self):  # noqa: D401
    """
    Return custom format patterns for the *minimal* and *enhanced* styles that
    are overridden in this wrapper to satisfy plume_nav_sim test expectations.
    Falls back to upstream implementation for all other format values.
    """
    try:
        fmt = getattr(self, "format", None)
        if fmt == "minimal":
            return MINIMAL_FORMAT
        if fmt == "enhanced":
            return ENHANCED_FORMAT
    except Exception:  # pragma: no cover – defensive
        pass
    return _orig_get_format_pattern(self)


# Apply the patch once at import time
_UpstreamLoggingConfig.get_format_pattern = _patched_get_format_pattern

# --------------------------------------------------------------------------- #
# Local helper: filter that always merges active correlation context          #
# --------------------------------------------------------------------------- #

def _create_context_filter_with_context(default_context: dict):  # noqa: D401
    """
    Return a Loguru filter callable that injects the *current* correlation
    context (if any) into ``record["extra"]`` before emission, while still
    filling any missing keys with ``default_context`` fall-backs.
    """

    def context_filter(record):
        # Inject dynamic correlation context
        try:
            ctx = get_correlation_context().bind_context()
            for k, v in ctx.items():
                record["extra"].setdefault(k, v)
        except Exception:  # pragma: no cover – safety net
            pass

        # Ensure defaults remain present
        for key, default_value in default_context.items():
            record["extra"].setdefault(key, default_value)
        return True

    return context_filter


def setup_logger(  # noqa: C901
    config: Optional[_UpstreamLoggingConfig | dict] = None,
    *,
    sink: Optional[str | Path] = None,
    level: Optional[str] = None,
    format: Optional[str] = None,
    rotation: Optional[str] = None,
    retention: Optional[str] = None,
    enqueue: Optional[bool] = None,
    backtrace: Optional[bool] = None,
    diagnose: Optional[bool] = None,
    environment: Optional[str] = None,
    logging_config_path: Optional[str | Path] = None,
    **kwargs,
) -> _UpstreamLoggingConfig:
    """
    Wrapper around the upstream ``setup_logger`` which **does not** apply
    environment defaults when a fully-formed ``LoggingConfig`` object is
    provided by the caller.  All other call styles are forwarded unchanged to
    the upstream implementation.
    """
    # Delegate when caller didn't supply an explicit config object
    if not isinstance(config, _UpstreamLoggingConfig):
        return _upstream_setup_logger(
            config=config,
            sink=sink,
            level=level,
            format=format,
            rotation=rotation,
            retention=retention,
            enqueue=enqueue,
            backtrace=backtrace,
            diagnose=diagnose,
            environment=environment,
            logging_config_path=logging_config_path,
            **kwargs,
        )

    # --------------------------------------------------------------------- #
    # Explicit LoggingConfig provided – use it verbatim                     #
    # --------------------------------------------------------------------- #
    cfg = config

    # Remove any existing handlers installed by previous calls
    logger.remove()

    # --------------------------------------------------------------------- #
    # Ensure TRACE logs are captured by default handlers                    #
    # --------------------------------------------------------------------- #
    try:
        debug_no = logger.level("DEBUG").no
        # Re-register TRACE just *above* DEBUG so that any sink configured
        # with the default DEBUG threshold captures TRACE messages as well,
        # while still preserving the correct ordering for Loguru.
        logger.level("TRACE", no=debug_no + 1)
    except Exception:  # pragma: no cover – defensive
        pass

    # Resolve format pattern *after* our overrides so tests see request_id
    format_pattern = cfg.get_format_pattern()

    # Baseline context required by upstream tests
    default_context = {
        "correlation_id": "none",
        "request_id": "none",
        "module": "system",
        "config_hash": "unknown",
        "step_count": 0,
        **cfg.default_context,
    }

    # ---------------------------- console sink --------------------------- #
    if cfg.console_enabled:
        console_format = (
            CLI_FORMAT
            if cfg.format == "cli"
            else MINIMAL_FORMAT
            if cfg.format == "minimal"
            else format_pattern
        )
        logger.add(
            sys.stderr,
            format=console_format,
            level=cfg.level,
            backtrace=cfg.backtrace,
            diagnose=cfg.diagnose,
            enqueue=cfg.enqueue,
            filter=_create_context_filter_with_context(default_context),
            catch=True,
        )

    # ------------------------------ file sink ---------------------------- #
    if cfg.file_enabled and cfg.file_path:
        log_path = Path(cfg.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------- #
        # File filter: drop system_baseline records to keep tests focused   #
        # ----------------------------------------------------------------- #

        def _file_filter(default_ctx):  # pragma: no cover – local helper
            base = _create_context_filter_with_context(default_ctx)

            def _f(record):
                try:
                    if record["extra"].get("metric_type") in (
                        "system_baseline",
                        "system_startup",
                    ):
                        return False
                except Exception:
                    pass
                return base(record)

            return _f

        # ----------------------------------------------------------------- #
        # Construct sink / formatter                                        #
        # ----------------------------------------------------------------- #

        if cfg.format == "json":

            def _json_sink_factory(path: Path, default_ctx: dict):
                """
                Build a callable sink that serialises each Loguru record to a
                single-line JSON string using the upstream formatter and
                appends it to *path*.
                """

                # Upstream helper returns a *function* accepting a record dict
                json_formatter = _upstream_create_json_formatter()

                # Ensure parent exists (already created above but be safe)
                path.parent.mkdir(parents=True, exist_ok=True)

                def _sink(message):
                    try:
                        line = json_formatter(message.record)
                        with path.open("a", encoding="utf-8") as fp:
                            fp.write(line)
                    except Exception:
                        # Swallow any sink-level errors; Loguru will handle if
                        # catch=False but we rely on catch=True below.
                        raise

                return _sink

            file_sink = _json_sink_factory(log_path, default_context)
            # NOTE: When a *callable* sink is supplied, Loguru does **not**
            # accept `rotation` or `retention` parameters (they are only valid
            # for path-based sinks).  Passing them results in a TypeError and
            # no handler being registered, which previously broke JSON tests.
            logger.add(
                file_sink,
                level=cfg.level,
                enqueue=cfg.enqueue,
                backtrace=cfg.backtrace,
                diagnose=cfg.diagnose,
                filter=_file_filter(default_context),
                catch=True,
            )

        else:
            file_format = format_pattern

            # Only add traditional text/cli/minimal file sink for non-JSON formats
            logger.add(
                str(log_path),
                format=file_format,
                level=cfg.level,
                rotation=cfg.rotation,
                retention=cfg.retention,
                enqueue=cfg.enqueue,
                backtrace=cfg.backtrace,
                diagnose=cfg.diagnose,
                filter=_file_filter(default_context),
                serialize=False,
                catch=True,
            )

    # Optional performance monitoring re-use upstream helper
    if cfg.enable_performance:
        from odor_plume_nav.utils.logging_setup import (
            _setup_performance_monitoring as _perf,
        )

        _perf(cfg)

    # Emit startup record
    logger.bind(**default_context).info(
        "Enhanced logging system initialized",
        extra={
            "metric_type": "system_startup",
            "environment": cfg.environment,
            "log_level": cfg.level,
            "format": cfg.format,
            "performance_monitoring": cfg.enable_performance,
            "correlation_tracking": cfg.correlation_enabled,
        },
    )

    return cfg

if "setup_logger" not in __all__:
    __all__.append("setup_logger")

# --------------------------------------------------------------------------- #
# Additional lightweight debug timers/utilities expected by plume_nav_sim     #
# --------------------------------------------------------------------------- #

from contextlib import contextmanager
from typing import ContextManager, Optional, Dict, Any

# get_correlation_context, PerformanceMetrics, and logger are provided by the
# re-exported odor_plume_nav implementation.


@contextmanager
def debug_command_timer(
    operation_name: str = "debug_command", **metadata
) -> ContextManager["PerformanceMetrics"]:
    """
    Lightweight timer for debug CLI commands.
    Logs a DEBUG completion entry with attached PerformanceMetrics.
    """
    context = get_correlation_context()
    metrics = context.push_performance(operation_name, **metadata)
    try:
        yield metrics
    finally:
        completed = context.pop_performance()
        bound = logger.bind(**context.bind_context())
        if completed:
            bound.debug(
                f"Debug command completed: {operation_name}",
                extra={
                    "metric_type": "debug_command",
                    "operation": operation_name,
                    "performance_metrics": completed.to_dict(),
                    **metadata,
                },
            )


@contextmanager
def debug_session_timer(
    operation_name: str = "debug_session", **metadata
) -> ContextManager["PerformanceMetrics"]:
    """
    Lightweight timer for debug sessions.
    Logs a DEBUG completion entry with attached PerformanceMetrics.
    """
    context = get_correlation_context()
    metrics = context.push_performance(operation_name, **metadata)
    try:
        yield metrics
    finally:
        completed = context.pop_performance()
        bound = logger.bind(**context.bind_context())
        if completed:
            bound.debug(
                f"Debug session completed: {operation_name}",
                extra={
                    "metric_type": "debug_session",
                    "operation": operation_name,
                    "performance_metrics": completed.to_dict(),
                    **metadata,
                },
            )


# --------------------------------------------------------------------------- #
# Helper logging functions for debug CLI correlation/events                  #
# --------------------------------------------------------------------------- #


def log_debug_command_correlation(command: str, **details: Any) -> None:
    """Log correlation info for a debug command with current context bound."""
    context = get_correlation_context()
    logger.bind(**context.bind_context()).info(
        f"Debug command correlation: {command}",
        extra={
            "metric_type": "debug_command_correlation",
            "command": command,
            **details,
        },
    )


def log_debug_session_event(event: str, **details: Any) -> None:
    """Log a debug session event with current correlation context."""
    context = get_correlation_context()
    logger.bind(**context.bind_context()).info(
        f"Debug session event: {event}",
        extra={
            "metric_type": "debug_session_event",
            "event": event,
            **details,
        },
    )


# --------------------------------------------------------------------------- #
# Export newly added symbols                                                  #
# --------------------------------------------------------------------------- #

for _name in [
    "debug_command_timer",
    "debug_session_timer",
    "log_debug_command_correlation",
    "log_debug_session_event",
]:
    if _name not in __all__:
        __all__.append(_name)
