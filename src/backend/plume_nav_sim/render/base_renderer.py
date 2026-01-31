"""Shared renderer base classes and helpers."""

import abc
import dataclasses
import time
import uuid
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .._compat import (
    ComponentError,
    RenderingError,
    ValidationError,
    validate_coordinates,
)
from ..constants import (
    FIELD_DTYPE,
    PERFORMANCE_TARGET_HUMAN_RENDER_MS,
    PERFORMANCE_TARGET_RGB_RENDER_MS,
    RGB_DTYPE,
    SUPPORTED_RENDER_MODES,
)
from ..core.types import Coordinates, GridSize, RenderMode, RGBArray
from ..logging import get_component_logger

DEFAULT_CONTEXT_VALIDATION = True
PERFORMANCE_MONITORING_ENABLED = True
RESOURCE_CLEANUP_TIMEOUT_SEC = 5.0

__all__ = [
    "BaseRenderer",
    "RenderContext",
    "RenderingMetrics",
    "create_render_context",
    "validate_rendering_parameters",
    "create_rendering_metrics",
]


@dataclasses.dataclass(frozen=True)
class RenderContext:

    concentration_field: np.ndarray
    agent_position: Coordinates
    source_position: Coordinates
    grid_size: GridSize

    context_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    creation_timestamp: float = dataclasses.field(default_factory=time.time)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def validate(  # noqa: C901
        self, strict_validation: bool = True, check_performance: bool = True
    ) -> bool:
        try:
            if not isinstance(self.concentration_field, np.ndarray):
                raise ValidationError(
                    "Concentration field must be NumPy array",
                    context={"actual_type": type(self.concentration_field).__name__},
                )

            if self.concentration_field.dtype != FIELD_DTYPE:
                raise ValidationError(
                    f"Concentration field must use {FIELD_DTYPE} dtype",
                    context={
                        "expected_dtype": str(FIELD_DTYPE),
                        "actual_dtype": str(self.concentration_field.dtype),
                    },
                )

            expected_shape = (self.grid_size.height, self.grid_size.width)
            if self.concentration_field.shape != expected_shape:
                raise ValidationError(
                    "Concentration field shape must match grid dimensions",
                    context={
                        "expected_shape": expected_shape,
                        "actual_shape": self.concentration_field.shape,
                        "grid_size": f"{self.grid_size.width}x{self.grid_size.height}",
                    },
                )

            min_concentration = float(np.min(self.concentration_field))
            max_concentration = float(np.max(self.concentration_field))

            if min_concentration < 0.0 or max_concentration > 1.0:
                if strict_validation:
                    raise ValidationError(
                        "Concentration values must be in range [0,1]",
                        context={
                            "min_value": min_concentration,
                            "max_value": max_concentration,
                            "expected_range": "[0.0, 1.0]",
                        },
                    )

            try:
                validate_coordinates(self.agent_position, self.grid_size)
            except ValidationError as exc:
                raise ValidationError(
                    "Agent position is outside grid boundaries",
                    context={
                        "agent_position": f"({self.agent_position.x}, {self.agent_position.y})",
                        "grid_bounds": f"0 <= x < {self.grid_size.width}, 0 <= y < {self.grid_size.height}",
                    },
                ) from exc

            try:
                validate_coordinates(self.source_position, self.grid_size)
            except ValidationError as exc:
                raise ValidationError(
                    "Source position is outside grid boundaries",
                    context={
                        "source_position": f"({self.source_position.x}, {self.source_position.y})",
                        "grid_bounds": f"0 <= x < {self.grid_size.width}, 0 <= y < {self.grid_size.height}",
                    },
                ) from exc

            if self.agent_position == self.source_position:
                pass

            if strict_validation:
                if not np.isfinite(self.concentration_field).all():
                    raise ValidationError(
                        "Concentration field contains NaN or infinite values",
                        context={
                            "nan_count": int(
                                np.sum(np.isnan(self.concentration_field))
                            ),
                            "inf_count": int(
                                np.sum(np.isinf(self.concentration_field))
                            ),
                        },
                    )

                try:
                    uuid.UUID(self.context_id)
                except ValueError:
                    raise ValidationError(
                        "Context ID must be valid UUID format",
                        context={"context_id": self.context_id},
                    )

            if check_performance:
                field_memory_mb = self.concentration_field.nbytes / (1024 * 1024)

                if field_memory_mb > 10.0:
                    pass

                if self.creation_timestamp <= 0:
                    raise ValidationError(
                        "Invalid creation timestamp",
                        context={"timestamp": self.creation_timestamp},
                    )

            return True

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Context validation failed with unexpected error: {e}",
                context={
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                    "context_id": getattr(self, "context_id", "unknown"),
                },
            )

    def get_context_summary(
        self, include_metadata: bool = True, include_performance_info: bool = True
    ) -> Dict[str, Any]:
        summary = {
            "context_id": self.context_id,
            "creation_timestamp": self.creation_timestamp,
            "grid_dimensions": f"{self.grid_size.width}x{self.grid_size.height}",
            "agent_position": f"({self.agent_position.x}, {self.agent_position.y})",
            "source_position": f"({self.source_position.x}, {self.source_position.y})",
        }

        field_stats = {
            "concentration_min": float(np.min(self.concentration_field)),
            "concentration_max": float(np.max(self.concentration_field)),
            "concentration_mean": float(np.mean(self.concentration_field)),
            "field_shape": self.concentration_field.shape,
            "field_dtype": str(self.concentration_field.dtype),
        }
        summary["concentration_stats"] = field_stats

        if include_metadata and self.metadata:
            summary["metadata"] = dict(self.metadata)

        if include_performance_info:
            performance_info = {
                "memory_usage_mb": round(
                    self.concentration_field.nbytes / (1024 * 1024), 3
                ),
                "total_grid_points": self.grid_size.width * self.grid_size.height,
                "age_seconds": round(time.time() - self.creation_timestamp, 3),
            }
            summary["performance_info"] = performance_info

        return summary

    def clone_with_overrides(
        self,
        agent_position: Optional[Coordinates] = None,
        source_position: Optional[Coordinates] = None,
        metadata: Optional[Dict[str, Any]] = None,
        new_agent_position: Optional[Coordinates] = None,
        new_source_position: Optional[Coordinates] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
    ) -> "RenderContext":
        resolved_agent = (
            agent_position if agent_position is not None else new_agent_position
        )
        resolved_source = (
            source_position if source_position is not None else new_source_position
        )

        agent_pos = (
            resolved_agent if resolved_agent is not None else self.agent_position
        )
        source_pos = (
            resolved_source if resolved_source is not None else self.source_position
        )

        updated_metadata = dict(self.metadata)
        meta_override = metadata if metadata is not None else new_metadata
        if meta_override is not None:
            updated_metadata.update(meta_override)

        new_context = RenderContext(
            concentration_field=self.concentration_field,
            agent_position=agent_pos,
            source_position=source_pos,
            grid_size=self.grid_size,
            context_id=str(uuid.uuid4()),
            creation_timestamp=time.time(),
            metadata=updated_metadata,
        )

        try:
            new_context.validate(strict_validation=True, check_performance=False)
        except ValidationError as e:
            raise ValidationError(
                f"Context cloning failed validation: {e}",
                context={
                    "original_context_id": self.context_id,
                    "clone_parameters": {
                        "agent_position": (
                            str(agent_position)
                            if agent_position is not None
                            else (
                                str(new_agent_position) if new_agent_position else None
                            )
                        ),
                        "source_position": (
                            str(source_position)
                            if source_position is not None
                            else (
                                str(new_source_position)
                                if new_source_position
                                else None
                            )
                        ),
                        "metadata_keys": (
                            list((metadata or new_metadata or {}).keys())
                            if (metadata is not None or new_metadata is not None)
                            else None
                        ),
                    },
                },
            )

        return new_context


@dataclasses.dataclass
class RenderingMetrics:

    renderer_type: str
    render_mode: RenderMode
    operation_id: str

    start_time: float = dataclasses.field(default_factory=time.time)
    end_time: Optional[float] = None
    timing_data: Dict[str, Any] = dataclasses.field(default_factory=dict)
    resource_usage: Dict[str, Any] = dataclasses.field(default_factory=dict)
    operation_count: int = 0
    performance_summary: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def record_rendering(
        self,
        duration_ms: float,
        operation_context: Optional[Dict[str, Any]] = None,
        validate_performance: bool = True,
    ) -> None:
        self.end_time = time.time()

        timing_entry = {
            "duration_ms": duration_ms,
            "completion_time": self.end_time,
            "render_mode": str(self.render_mode),
            "renderer_type": self.renderer_type,
        }

        if operation_context:
            timing_entry["context"] = dict(operation_context)

        operation_key = f"operation_{self.operation_count}"
        self.timing_data[operation_key] = timing_entry

        self.operation_count += 1

        if validate_performance:
            target_ms = self._get_performance_target()
            if duration_ms > target_ms:
                warning_msg = (
                    f"Rendering exceeded target: {duration_ms:.2f}ms > {target_ms}ms"
                )
                self.performance_summary.setdefault("warnings", []).append(
                    {
                        "timestamp": self.end_time,
                        "message": warning_msg,
                        "duration_ms": duration_ms,
                        "target_ms": target_ms,
                    }
                )

        self._update_performance_summary(duration_ms)

    def check_performance_targets(  # noqa: C901
        self, strict_checking: bool = False, generate_recommendations: bool = True
    ) -> Dict[str, Any]:
        target_ms = self._get_performance_target()
        analysis_result = {
            "target_compliance": True,
            "performance_target_ms": target_ms,
            "render_mode": str(self.render_mode),
            "analysis_timestamp": time.time(),
        }

        if self.timing_data:
            durations = [
                entry["duration_ms"]
                for entry in self.timing_data.values()
                if isinstance(entry, dict) and "duration_ms" in entry
            ]

            if durations:
                latest_duration = durations[-1]
                average_duration = sum(durations) / len(durations)
                max_duration = max(durations)

                tolerance_factor = 1.1 if strict_checking else 1.5
                effective_target = target_ms * tolerance_factor

                analysis_result.update(
                    {
                        "latest_duration_ms": latest_duration,
                        "average_duration_ms": round(average_duration, 2),
                        "max_duration_ms": max_duration,
                        "sample_count": len(durations),
                        "target_compliance": latest_duration <= effective_target,
                    }
                )

                if max_duration > target_ms * 2:
                    analysis_result["bottleneck_detected"] = True
                    analysis_result["bottleneck_severity"] = (
                        "high" if max_duration > target_ms * 5 else "moderate"
                    )

                target_ratio = average_duration / target_ms
                analysis_result["performance_ratio"] = round(target_ratio, 2)
                analysis_result["efficiency_percentage"] = round(
                    min(100, target_ms / average_duration * 100), 1
                )

        if generate_recommendations:
            recommendations = []

            if not analysis_result["target_compliance"]:
                recommendations.append(
                    "Performance target missed - consider optimization strategies"
                )

            if analysis_result.get("bottleneck_detected"):
                recommendations.append(
                    "Performance bottleneck detected - investigate resource usage"
                )

            if self.operation_count > 0:
                avg_duration = analysis_result.get("average_duration_ms", 0)
                if avg_duration > target_ms:
                    if self.render_mode == RenderMode.RGB_ARRAY:
                        recommendations.append(
                            "RGB rendering slow - consider array optimization or caching"
                        )
                    elif self.render_mode == RenderMode.HUMAN:
                        recommendations.append(
                            "Human rendering slow - consider reducing update frequency"
                        )

            if not recommendations:
                recommendations.append("Performance within acceptable limits")

            analysis_result["recommendations"] = recommendations

        return analysis_result

    def get_performance_summary(
        self, include_history: bool = True, include_recommendations: bool = True
    ) -> Dict[str, Any]:
        summary = {
            "renderer_type": self.renderer_type,
            "render_mode": str(self.render_mode),
            "operation_id": self.operation_id,
            "total_operations": self.operation_count,
            "analysis_timestamp": time.time(),
        }

        if self.timing_data:
            durations = [
                entry["duration_ms"]
                for entry in self.timing_data.values()
                if isinstance(entry, dict) and "duration_ms" in entry
            ]

            if durations:
                statistics = {
                    "average_duration_ms": round(sum(durations) / len(durations), 3),
                    "min_duration_ms": round(min(durations), 3),
                    "max_duration_ms": round(max(durations), 3),
                    "total_render_time_ms": round(sum(durations), 3),
                    "operations_per_second": (
                        round(len(durations) / (sum(durations) / 1000), 2)
                        if sum(durations) > 0
                        else 0
                    ),
                }
                summary["operation_statistics"] = statistics

        if include_history and self.timing_data:
            history_data = {
                "timing_entries": len(self.timing_data),
                "oldest_entry": min(
                    (
                        entry.get("completion_time", 0)
                        for entry in self.timing_data.values()
                        if isinstance(entry, dict)
                    ),
                    default=0,
                ),
                "newest_entry": max(
                    (
                        entry.get("completion_time", 0)
                        for entry in self.timing_data.values()
                        if isinstance(entry, dict)
                    ),
                    default=0,
                ),
            }
            summary["historical_data"] = history_data

        if self.resource_usage:
            summary["resource_metrics"] = dict(self.resource_usage)

        if self.performance_summary:
            summary["performance_summary"] = dict(self.performance_summary)

        if include_recommendations:
            target_analysis = self.check_performance_targets(
                strict_checking=False, generate_recommendations=True
            )
            summary["optimization_analysis"] = target_analysis

        return summary

    def reset_metrics(
        self, preserve_history: bool = False, new_operation_id: Optional[str] = None
    ) -> None:
        if preserve_history and (self.timing_data or self.resource_usage):
            history_entry = {
                "preserved_timestamp": time.time(),
                "operation_count": self.operation_count,
                "timing_data_entries": len(self.timing_data),
                "resource_entries": len(self.resource_usage),
                "operation_duration_sec": (
                    (self.end_time - self.start_time) if self.end_time else None
                ),
            }

            self.performance_summary.setdefault("history", []).append(history_entry)

        self.timing_data.clear()
        self.resource_usage.clear()
        self.operation_count = 0

        if new_operation_id is not None:
            self.operation_id = new_operation_id

        self.start_time = time.time()
        self.end_time = None

        if "warnings" in self.performance_summary and not preserve_history:
            self.performance_summary["warnings"].clear()

    def _get_performance_target(self) -> float:
        if self.render_mode == RenderMode.RGB_ARRAY:
            return PERFORMANCE_TARGET_RGB_RENDER_MS
        elif self.render_mode == RenderMode.HUMAN:
            return PERFORMANCE_TARGET_HUMAN_RENDER_MS
        else:
            return PERFORMANCE_TARGET_RGB_RENDER_MS

    def _update_performance_summary(self, latest_duration_ms: float) -> None:
        if "total_duration_ms" not in self.performance_summary:
            self.performance_summary["total_duration_ms"] = 0.0
            self.performance_summary["operation_count"] = 0

        self.performance_summary["total_duration_ms"] += latest_duration_ms
        self.performance_summary["operation_count"] = self.operation_count

        if self.operation_count > 0:
            self.performance_summary["average_duration_ms"] = round(
                self.performance_summary["total_duration_ms"] / self.operation_count, 3
            )

        self.performance_summary["last_update"] = time.time()
        self.performance_summary["latest_duration_ms"] = latest_duration_ms


class BaseRenderer(abc.ABC):

    def __init__(
        self,
        grid_size: GridSize,
        color_scheme_name: Optional[str] = None,
        renderer_options: Optional[Dict[str, Any]] = None,
    ):
        if not isinstance(grid_size, GridSize):
            raise ValidationError(
                "Grid size must be GridSize instance",
                context={"provided_type": type(grid_size).__name__},
            )
        if grid_size.width <= 0 or grid_size.height <= 0:
            raise ValidationError(
                "Grid dimensions must be positive (no negative or zero values)",
                context={
                    "width": grid_size.width,
                    "height": grid_size.height,
                },
            )
        self.grid_size = grid_size

        self.color_scheme_name = color_scheme_name

        self.renderer_options = renderer_options or {}

        self.logger = get_component_logger("render")

        self._initialized = False

        self._current_metrics: Optional[RenderingMetrics] = None

        self._resource_cache: Dict[str, Any] = {}

        self._last_warning_time: Dict[str, float] = {}
        self._warning_throttle_seconds = (
            renderer_options.get("warning_throttle_seconds", 10.0)
            if renderer_options
            else 10.0
        )
        self._mute_perf_warnings = (
            renderer_options.get("mute_performance_warnings", False)
            if renderer_options
            else False
        )

        self.logger.info(
            f"BaseRenderer initialized with grid_size={grid_size.width}x{grid_size.height}, "
            f"color_scheme={color_scheme_name}, options={len(self.renderer_options)} parameters"
        )

    def initialize(  # noqa: C901
        self,
        validate_immediately: bool = DEFAULT_CONTEXT_VALIDATION,
        enable_performance_monitoring: bool = PERFORMANCE_MONITORING_ENABLED,
    ) -> None:
        try:
            if (
                self.color_scheme_name is not None
                and not str(self.color_scheme_name).strip()
            ):
                raise ValidationError(
                    "Invalid color_scheme_name: cannot be empty",
                    context={"color_scheme_name": self.color_scheme_name},
                )

            if self.grid_size.width <= 0 or self.grid_size.height <= 0:
                raise ValidationError(
                    "Grid dimensions must be positive",
                    context={
                        "grid_size": f"{self.grid_size.width}x{self.grid_size.height}"
                    },
                )

            estimated_memory_mb = (self.grid_size.width * self.grid_size.height * 4) / (
                1024 * 1024
            )
            if estimated_memory_mb > 100:
                self.logger.warning(
                    f"Large grid may impact performance: {estimated_memory_mb:.1f}MB estimated"
                )

            self._initialize_renderer_resources()

            if enable_performance_monitoring:
                self.logger.info(
                    "Performance monitoring enabled for rendering operations"
                )

            self._initialized = True

            if validate_immediately:
                test_field = np.zeros(
                    (self.grid_size.height, self.grid_size.width), dtype=FIELD_DTYPE
                )
                test_context = RenderContext(
                    concentration_field=test_field,
                    agent_position=Coordinates(x=0, y=0),
                    source_position=Coordinates(
                        x=self.grid_size.width // 2, y=self.grid_size.height // 2
                    ),
                    grid_size=self.grid_size,
                )
                self.validate_context(test_context, strict_validation=False)

            self.logger.info(
                f"Renderer initialization completed: grid={self.grid_size.width}x{self.grid_size.height}, "
                f"performance_monitoring={enable_performance_monitoring}, "
                f"immediate_validation={validate_immediately}"
            )

        except Exception as e:
            self.logger.error(f"Renderer initialization failed: {e}")
            self._initialized = False

            try:
                self._cleanup_renderer_resources()
            except Exception as cleanup_error:
                self.logger.error(
                    f"Cleanup after initialization failure failed: {cleanup_error}"
                )

            if isinstance(e, (ValidationError, ComponentError, RenderingError)):
                raise
            else:
                raise RenderingError(f"Renderer initialization failed: {e}")

    def render(  # noqa: C901
        self, context: RenderContext, mode_override: Optional[RenderMode] = None
    ) -> Union[RGBArray, None]:
        if not self._initialized:
            try:
                self.initialize(validate_immediately=False)
            except Exception:
                pass

        self._check_initialization()

        try:
            self.validate_context(context, strict_validation=False)
        except ValidationError as ve:
            raise RenderingError(
                f"Rendering operation failed: {ve}",
                context={
                    "reason": "context_validation_failed",
                    "renderer": self.__class__.__name__,
                },
            )

        effective_mode = mode_override
        if effective_mode is None:
            for mode in [RenderMode.RGB_ARRAY, RenderMode.HUMAN]:
                if self.supports_render_mode(mode):
                    effective_mode = mode
                    break

        if effective_mode is None:
            raise RenderingError("No supported render modes available")

        if not self.supports_render_mode(effective_mode):
            raise ComponentError(
                f"Render mode not supported: {effective_mode}",
                component_name=self.__class__.__name__,
                operation_name="render",
            )

        track_perf = True
        if hasattr(self, "performance_config") and isinstance(
            getattr(self, "performance_config", None), dict
        ):
            track_perf = self.performance_config.get("track_render_performance", True)

        if track_perf:
            operation_context = {
                "render_mode": str(effective_mode),
                "grid_size": f"{self.grid_size.width}x{self.grid_size.height}",
                "context_id": context.context_id,
            }
            self._start_performance_tracking("render", operation_context)

        try:
            start_time = time.time()

            if effective_mode == RenderMode.RGB_ARRAY:
                result = self._render_rgb_array(context)

                if not isinstance(result, np.ndarray):
                    raise RenderingError("RGB renderer must return numpy array")
                if result.dtype != RGB_DTYPE:
                    raise RenderingError(f"RGB array must use {RGB_DTYPE} dtype")
                if len(result.shape) != 3 or result.shape[2] != 3:
                    raise RenderingError("RGB array must have shape (H,W,3)")

            elif effective_mode == RenderMode.HUMAN:
                result = self._render_human(context)
                if result is not None:
                    self.logger.warning("Human mode renderer returned non-None value")
                result = None

            else:
                raise RenderingError(f"Unsupported render mode: {effective_mode}")

            duration_ms = (time.time() - start_time) * 1000
            if track_perf:
                self._record_performance("render", duration_ms, check_targets=True)

            return result

        except Exception as e:
            duration_ms = (
                (time.time() - start_time) * 1000 if "start_time" in locals() else 0
            )

            if self._current_metrics:
                self._current_metrics.performance_summary.setdefault(
                    "errors", []
                ).append(
                    {
                        "timestamp": time.time(),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "duration_ms": duration_ms,
                        "render_mode": str(effective_mode),
                    }
                )

            self.logger.error(
                f"Rendering failed: {e}, mode={effective_mode}, "
                f"context_id={context.context_id}, duration={duration_ms:.2f}ms"
            )

            if isinstance(e, RenderingError):
                raise
            else:
                raise RenderingError(
                    f"Rendering operation failed: {e}", context=operation_context
                )

    def cleanup_resources(
        self,
        timeout: Optional[float] = None,
        force: Optional[bool] = None,
        timeout_sec: float = RESOURCE_CLEANUP_TIMEOUT_SEC,
        force_cleanup: bool = False,
    ) -> bool:
        cleanup_start_time = time.time()
        cleanup_successful = True

        try:
            effective_timeout = (
                timeout
                if (timeout is not None and timeout > 0)
                else (timeout_sec if timeout_sec > 0 else RESOURCE_CLEANUP_TIMEOUT_SEC)
            )
            effective_force = force if (force is not None) else force_cleanup

            self.logger.info(
                f"Starting resource cleanup with timeout={effective_timeout}s, force={effective_force}"
            )

            try:
                self._cleanup_renderer_resources()
            except Exception as e:
                cleanup_successful = False
                if effective_force:
                    self.logger.warning(
                        f"Forced cleanup ignoring implementation error: {e}"
                    )
                else:
                    self.logger.error(f"Implementation cleanup failed: {e}")
                    return False

            if self._resource_cache:
                cache_size = len(self._resource_cache)
                self._resource_cache.clear()
                self.logger.debug(f"Cleared resource cache: {cache_size} entries")

            if self._current_metrics:
                self._current_metrics.reset_metrics(preserve_history=False)
                self._current_metrics = None

            self._initialized = False

            cleanup_duration = time.time() - cleanup_start_time

            self.logger.info(
                f"Resource cleanup completed: success={cleanup_successful}, "
                f"duration={cleanup_duration:.3f}s, forced={effective_force}"
            )

            if cleanup_duration > effective_timeout:
                self.logger.warning(
                    f"Cleanup exceeded timeout: {cleanup_duration:.3f}s > {effective_timeout}s"
                )

            return cleanup_successful

        except Exception as e:
            cleanup_duration = time.time() - cleanup_start_time
            self.logger.error(
                f"Resource cleanup failed with exception: {e}, "
                f"duration={cleanup_duration:.3f}s, force={force_cleanup}"
            )

            if force_cleanup:
                self._initialized = False
                self._resource_cache.clear()
                self._current_metrics = None
                return True

            return False

    @abc.abstractmethod
    def supports_render_mode(self, mode: RenderMode) -> bool:
        pass

    def get_performance_metrics(
        self, reset_after_retrieval: bool = False, include_system_metrics: bool = False
    ) -> Dict[str, Any]:
        metrics_data = {
            "collection_timestamp": time.time(),
            "renderer_type": self.__class__.__name__,
            "grid_size": f"{self.grid_size.width}x{self.grid_size.height}",
            "initialized": self._initialized,
        }

        if self._current_metrics:
            performance_summary = self._current_metrics.get_performance_summary(
                include_history=True, include_recommendations=True
            )
            metrics_data["performance_data"] = performance_summary

            metrics_data["render_operations"] = self._current_metrics.operation_count
            metrics_data["resource_usage"] = dict(self._current_metrics.resource_usage)

            target_analysis = self._current_metrics.check_performance_targets(
                strict_checking=False, generate_recommendations=True
            )
            metrics_data["target_analysis"] = target_analysis

        else:
            metrics_data["performance_data"] = {"status": "no_metrics_available"}
            metrics_data["render_operations"] = 0
            metrics_data["resource_usage"] = {}

        if include_system_metrics:
            system_info = {
                "resource_cache_size": len(self._resource_cache),
                "color_scheme": self.color_scheme_name,
                "renderer_options_count": len(self.renderer_options),
                "supported_modes": [
                    str(mode)
                    for mode in [RenderMode.RGB_ARRAY, RenderMode.HUMAN]
                    if self.supports_render_mode(mode)
                ],
            }
            metrics_data["system_metrics"] = system_info

        recommendations = []
        if self._current_metrics and hasattr(
            self._current_metrics, "performance_summary"
        ):
            if self._current_metrics.operation_count > 10:
                avg_duration = self._current_metrics.performance_summary.get(
                    "average_duration_ms", 0
                )
                target_ms = self._current_metrics._get_performance_target()

                if avg_duration > target_ms * 1.5:
                    recommendations.append(
                        "Consider performance optimization - operations consistently slow"
                    )
                elif avg_duration < target_ms * 0.1:
                    recommendations.append(
                        "Excellent performance - consider more complex visualizations"
                    )

        if not recommendations:
            recommendations.append("Performance within expected parameters")

        metrics_data["optimization_recommendations"] = recommendations

        if reset_after_retrieval and self._current_metrics:
            self._current_metrics.reset_metrics(preserve_history=True)
            self.logger.debug("Performance metrics reset after retrieval")

        return metrics_data

    def validate_context(
        self, context: RenderContext, strict_validation: bool = True
    ) -> bool:
        try:
            context.validate(
                strict_validation=strict_validation, check_performance=True
            )

            if context.grid_size != self.grid_size:
                raise ValidationError(
                    "Context grid size does not match renderer configuration",
                    context={
                        "context_grid_size": f"{context.grid_size.width}x{context.grid_size.height}",
                        "renderer_grid_size": f"{self.grid_size.width}x{self.grid_size.height}",
                        "suggestion": "Recreate renderer with matching grid size or update context",
                    },
                )

            if context.concentration_field.shape != (
                self.grid_size.height,
                self.grid_size.width,
            ):
                raise ValidationError(
                    "Concentration field shape inconsistent with renderer grid configuration",
                    context={
                        "field_shape": context.concentration_field.shape,
                        "expected_shape": (self.grid_size.height, self.grid_size.width),
                        "renderer_config": f"{self.grid_size.width}x{self.grid_size.height}",
                    },
                )

            if strict_validation:
                field_memory_mb = context.concentration_field.nbytes / (1024 * 1024)
                if field_memory_mb > 50:
                    self.logger.warning(
                        f"Large concentration field may impact performance: {field_memory_mb:.1f}MB"
                    )

                time_since_creation = time.time() - context.creation_timestamp
                if time_since_creation < 0 or time_since_creation > 3600:
                    raise ValidationError(
                        "Context creation timestamp appears invalid",
                        context={
                            "creation_timestamp": context.creation_timestamp,
                            "time_since_creation_sec": time_since_creation,
                            "current_timestamp": time.time(),
                        },
                    )

            if not self._initialized:
                raise ValidationError(
                    "Cannot validate context - renderer not initialized",
                    context={"renderer_state": "uninitialized"},
                )

            return True

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Context validation failed with unexpected error: {e}",
                context={
                    "error_type": type(e).__name__,
                    "context_id": getattr(context, "context_id", "unknown"),
                    "renderer_type": self.__class__.__name__,
                    "validation_mode": "strict" if strict_validation else "normal",
                },
            )

    @abc.abstractmethod
    def _initialize_renderer_resources(self) -> None:
        pass

    @abc.abstractmethod
    def _cleanup_renderer_resources(self) -> None:
        pass

    @abc.abstractmethod
    def _render_rgb_array(self, context: RenderContext) -> RGBArray:
        pass

    @abc.abstractmethod
    def _render_human(self, context: RenderContext) -> None:
        pass

    def _start_performance_tracking(
        self, operation_type: str, operation_context: Optional[Dict[str, Any]] = None
    ) -> None:
        if self._current_metrics is None:
            self._current_metrics = RenderingMetrics(
                renderer_type=self.__class__.__name__,
                render_mode=RenderMode.RGB_ARRAY,
                operation_id=str(uuid.uuid4()),
            )

        if operation_context and "render_mode" in operation_context:
            mode_str = operation_context["render_mode"]
            try:
                if mode_str == "rgb_array":
                    self._current_metrics.render_mode = RenderMode.RGB_ARRAY
                elif mode_str == "human":
                    self._current_metrics.render_mode = RenderMode.HUMAN
            except Exception as e:
                self.logger.debug(f"Failed to update render mode in metrics: {e}")

        self._current_metrics.start_time = time.time()
        self._current_metrics.end_time = None

        if operation_context:
            context_key = f"{operation_type}_{int(time.time() * 1000)}"
            self._current_metrics.resource_usage[context_key] = {
                "operation_type": operation_type,
                "context": dict(operation_context),
                "start_timestamp": self._current_metrics.start_time,
            }

    def _record_performance(
        self, operation_type: str, duration_ms: float, check_targets: bool = True
    ) -> None:
        if self._current_metrics is None:
            self._current_metrics = RenderingMetrics(
                renderer_type=self.__class__.__name__,
                render_mode=RenderMode.RGB_ARRAY,
                operation_id=str(uuid.uuid4()),
            )

        operation_context = {
            "operation_type": operation_type,
            "renderer_class": self.__class__.__name__,
            "grid_size": f"{self.grid_size.width}x{self.grid_size.height}",
        }

        self._current_metrics.record_rendering(
            duration_ms=duration_ms,
            operation_context=operation_context,
            validate_performance=check_targets,
        )

        if check_targets:
            target_ms = self._current_metrics._get_performance_target()

            if duration_ms > target_ms:
                if self._should_emit_warning(f"perf_target_{operation_type}"):
                    performance_ratio = duration_ms / target_ms
                    self.logger.warning(
                        f"Operation exceeded performance target: {duration_ms:.2f}ms > {target_ms}ms "
                        f"({performance_ratio:.1f}x slower), operation={operation_type}"
                    )

                    if performance_ratio > 5.0:
                        self.logger.warning(
                            "Severe performance issue detected - investigate optimization opportunities"
                        )
                    elif performance_ratio > 2.0:
                        self.logger.info(
                            "Moderate performance issue - consider optimization"
                        )

    def _should_emit_warning(self, warning_key: str) -> bool:
        if self._mute_perf_warnings:
            return False

        current_time = time.time()
        last_time = self._last_warning_time.get(warning_key, 0.0)

        if current_time - last_time >= self._warning_throttle_seconds:
            self._last_warning_time[warning_key] = current_time
            return True
        return False

    def _check_initialization(self) -> None:
        if not self._initialized:
            error_message = (
                f"Renderer {self.__class__.__name__} is not initialized. "
                "Call initialize() before using rendering operations."
            )

            self.logger.error(
                f"Operation attempted on uninitialized renderer: {self.__class__.__name__}"
            )

            raise ComponentError(
                error_message,
                component_name=self.__class__.__name__,
                operation_name="render",
            )


def create_render_context(  # noqa: C901
    concentration_field: np.ndarray,
    agent_position: Coordinates,
    source_position: Coordinates,
    grid_size: GridSize,
    context_id: Optional[str] = None,
    validate_immediately: bool = False,
) -> RenderContext:
    try:
        if not isinstance(concentration_field, np.ndarray):
            raise ValidationError(
                "Concentration field must be NumPy array",
                context={"provided_type": type(concentration_field).__name__},
            )

        if concentration_field.dtype != FIELD_DTYPE:
            try:
                concentration_field = concentration_field.astype(
                    FIELD_DTYPE, copy=False
                )
            except Exception as e:
                raise ValidationError(
                    f"Failed to convert concentration field to {FIELD_DTYPE}: {e}",
                    context={
                        "expected_dtype": str(FIELD_DTYPE),
                        "actual_dtype": str(concentration_field.dtype),
                    },
                )

        if context_id is None:
            context_id = str(uuid.uuid4())
        else:
            if not isinstance(context_id, str) or len(context_id.strip()) == 0:
                raise ValidationError(
                    "Context ID must be non-empty string",
                    context={"provided_context_id": repr(context_id)},
                )

        context = RenderContext(
            concentration_field=concentration_field,
            agent_position=agent_position,
            source_position=source_position,
            grid_size=grid_size,
            context_id=context_id.strip(),
            metadata={},
        )

        if validate_immediately:
            context.validate(strict_validation=True, check_performance=True)

        logger = get_component_logger("render")
        logger.debug(
            f"RenderContext created: id={context_id[:8]}..., "
            f"grid={grid_size.width}x{grid_size.height}, "
            f"agent=({agent_position.x},{agent_position.y}), "
            f"source=({source_position.x},{source_position.y}), "
            f"validated={validate_immediately}"
        )

        return context

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Context creation failed with unexpected error: {e}",
            context={
                "error_type": type(e).__name__,
                "original_error": str(e),
                "function": "create_render_context",
            },
        )


def validate_rendering_parameters(  # noqa: C901
    render_mode: RenderMode,
    grid_size: GridSize,
    color_scheme_name: Optional[str] = None,
    check_performance_feasibility: bool = True,
    strict_validation: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    validation_start_time = time.time()

    validation_report = {
        "validation_timestamp": validation_start_time,
        "parameters_tested": {
            "render_mode": str(render_mode),
            "grid_size": f"{grid_size.width}x{grid_size.height}",
            "color_scheme": color_scheme_name,
            "performance_check": check_performance_feasibility,
            "strict_mode": strict_validation,
        },
        "validation_results": {},
        "warnings": [],
        "recommendations": [],
        "performance_analysis": {},
        "system_compatibility": {},
    }

    is_valid = True

    try:
        if render_mode not in [RenderMode.RGB_ARRAY, RenderMode.HUMAN]:
            is_valid = False
            validation_report["validation_results"]["render_mode"] = {
                "valid": False,
                "error": f"Unsupported render mode: {render_mode}",
                "supported_modes": [
                    str(mode) for mode in [RenderMode.RGB_ARRAY, RenderMode.HUMAN]
                ],
            }
        else:
            mode_str = str(render_mode)
            if mode_str in SUPPORTED_RENDER_MODES:
                validation_report["validation_results"]["render_mode"] = {
                    "valid": True,
                    "mode": mode_str,
                    "programmatic": render_mode.is_programmatic(),
                    "requires_display": render_mode.requires_display(),
                }
            else:
                is_valid = False
                validation_report["validation_results"]["render_mode"] = {
                    "valid": False,
                    "error": f"Render mode not in supported list: {SUPPORTED_RENDER_MODES}",
                }

        grid_validation = {
            "valid": True,
            "width": grid_size.width,
            "height": grid_size.height,
            "total_points": grid_size.width * grid_size.height,
        }

        if grid_size.width <= 0 or grid_size.height <= 0:
            is_valid = False
            grid_validation["valid"] = False
            grid_validation["error"] = "Grid dimensions must be positive"
        elif grid_size.width > 2048 or grid_size.height > 2048:
            if strict_validation:
                is_valid = False
                grid_validation["valid"] = False
                grid_validation["error"] = (
                    "Grid dimensions too large for reliable performance"
                )
            else:
                validation_report["warnings"].append(
                    "Large grid dimensions may impact performance"
                )

        validation_report["validation_results"]["grid_size"] = grid_validation

        if color_scheme_name is not None:
            color_scheme_validation = {"valid": True, "scheme_name": color_scheme_name}

            if (
                not isinstance(color_scheme_name, str)
                or len(color_scheme_name.strip()) == 0
            ):
                is_valid = False
                color_scheme_validation["valid"] = False
                color_scheme_validation["error"] = (
                    "Color scheme name must be non-empty string"
                )
            else:
                color_scheme_validation["note"] = (
                    "Color scheme availability depends on renderer implementation"
                )

            validation_report["validation_results"][
                "color_scheme"
            ] = color_scheme_validation

        if check_performance_feasibility:
            performance_analysis = {}

            total_points = grid_size.width * grid_size.height
            field_memory_mb = (total_points * 4) / (1024 * 1024)

            if render_mode == RenderMode.RGB_ARRAY:
                rgb_memory_mb = (total_points * 3) / (1024 * 1024)
                performance_analysis["memory_estimates"] = {
                    "concentration_field_mb": round(field_memory_mb, 2),
                    "rgb_output_mb": round(rgb_memory_mb, 2),
                    "total_estimated_mb": round(field_memory_mb + rgb_memory_mb, 2),
                }

                estimated_rgb_time_ms = 0.001 * total_points / 1000
                performance_analysis["timing_estimates"] = {
                    "estimated_rgb_render_ms": round(estimated_rgb_time_ms, 2),
                    "target_rgb_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
                    "within_target": estimated_rgb_time_ms
                    <= PERFORMANCE_TARGET_RGB_RENDER_MS,
                }

            elif render_mode == RenderMode.HUMAN:
                performance_analysis["memory_estimates"] = {
                    "concentration_field_mb": round(field_memory_mb, 2),
                    "matplotlib_overhead_mb": 5.0,
                }

                estimated_human_time_ms = 0.01 * total_points / 1000
                performance_analysis["timing_estimates"] = {
                    "estimated_human_render_ms": round(estimated_human_time_ms, 2),
                    "target_human_ms": PERFORMANCE_TARGET_HUMAN_RENDER_MS,
                    "within_target": estimated_human_time_ms
                    <= PERFORMANCE_TARGET_HUMAN_RENDER_MS,
                }

            validation_report["performance_analysis"] = performance_analysis

            if render_mode == RenderMode.RGB_ARRAY:
                if (
                    performance_analysis["timing_estimates"]["estimated_rgb_render_ms"]
                    > PERFORMANCE_TARGET_RGB_RENDER_MS
                ):
                    validation_report["warnings"].append(
                        f"RGB rendering may exceed {PERFORMANCE_TARGET_RGB_RENDER_MS}ms target"
                    )
            elif render_mode == RenderMode.HUMAN:
                if (
                    performance_analysis["timing_estimates"][
                        "estimated_human_render_ms"
                    ]
                    > PERFORMANCE_TARGET_HUMAN_RENDER_MS
                ):
                    validation_report["warnings"].append(
                        f"Human rendering may exceed {PERFORMANCE_TARGET_HUMAN_RENDER_MS}ms target"
                    )

        if strict_validation:
            if grid_size.width != grid_size.height:
                validation_report["warnings"].append(
                    "Non-square grids may have different performance characteristics"
                )

            if grid_size.width % 8 != 0 or grid_size.height % 8 != 0:
                validation_report["warnings"].append(
                    "Grid dimensions not divisible by 8 may impact vectorization"
                )

        if is_valid:
            validation_report["recommendations"].append(
                "All validation checks passed successfully"
            )

        if (
            check_performance_feasibility
            and "performance_analysis" in validation_report
        ):
            memory_mb = validation_report["performance_analysis"][
                "memory_estimates"
            ].get("total_estimated_mb", 0)

            if memory_mb > 100:
                validation_report["recommendations"].append(
                    "High memory usage - consider smaller grid or optimization"
                )
            elif memory_mb < 1:
                validation_report["recommendations"].append(
                    "Low memory usage - grid size is very efficient"
                )

            if render_mode == RenderMode.RGB_ARRAY:
                validation_report["recommendations"].append(
                    "RGB mode selected - optimized for programmatic analysis"
                )
            elif render_mode == RenderMode.HUMAN:
                validation_report["recommendations"].append(
                    "Human mode selected - optimized for interactive visualization"
                )

        validation_duration_ms = (time.time() - validation_start_time) * 1000
        validation_report["validation_duration_ms"] = round(validation_duration_ms, 3)

        return (is_valid, validation_report)

    except Exception as e:
        validation_report["validation_results"]["error"] = {
            "exception_type": type(e).__name__,
            "error_message": str(e),
            "validation_failed": True,
        }
        validation_report["recommendations"] = [
            "Fix validation error before proceeding with renderer creation"
        ]

        return (False, validation_report)


def create_rendering_metrics(
    renderer_type: str,
    render_mode: RenderMode,
    operation_id: Optional[str] = None,
    enable_detailed_tracking: bool = True,
) -> RenderingMetrics:
    try:
        if not isinstance(renderer_type, str) or len(renderer_type.strip()) == 0:
            raise ValidationError(
                "Renderer type must be non-empty string",
                context={"provided_type": repr(renderer_type)},
            )

        if not isinstance(render_mode, RenderMode):
            raise ValidationError(
                "Render mode must be RenderMode enumeration",
                context={"provided_type": type(render_mode).__name__},
            )

        if operation_id is None:
            operation_id = str(uuid.uuid4())
        else:
            if not isinstance(operation_id, str) or len(operation_id.strip()) == 0:
                raise ValidationError(
                    "Operation ID must be non-empty string",
                    context={"provided_id": repr(operation_id)},
                )
            operation_id = operation_id.strip()

        metrics = RenderingMetrics(
            renderer_type=renderer_type.strip(),
            render_mode=render_mode,
            operation_id=operation_id,
        )

        if enable_detailed_tracking:
            metrics.resource_usage["detailed_tracking"] = {
                "enabled": True,
                "initialization_time": time.time(),
                "tracking_features": [
                    "timing_analysis",
                    "resource_monitoring",
                    "performance_validation",
                    "optimization_recommendations",
                ],
            }

        target_ms = (
            PERFORMANCE_TARGET_RGB_RENDER_MS
            if render_mode == RenderMode.RGB_ARRAY
            else PERFORMANCE_TARGET_HUMAN_RENDER_MS
        )

        metrics.performance_summary["performance_target_ms"] = target_ms
        metrics.performance_summary["render_mode"] = str(render_mode)
        metrics.performance_summary["detailed_tracking"] = enable_detailed_tracking

        metrics.timing_data["initialization"] = {
            "creation_timestamp": time.time(),
            "renderer_type": renderer_type,
            "render_mode": str(render_mode),
            "tracking_enabled": enable_detailed_tracking,
        }

        logger = get_component_logger("render")
        logger.debug(
            f"RenderingMetrics created: type={renderer_type}, mode={render_mode}, "
            f"id={operation_id[:8]}..., detailed_tracking={enable_detailed_tracking}"
        )

        return metrics

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Metrics creation failed with unexpected error: {e}",
            context={
                "error_type": type(e).__name__,
                "original_error": str(e),
                "function": "create_rendering_metrics",
                "renderer_type": (
                    renderer_type if "renderer_type" in locals() else "unknown"
                ),
            },
        )
