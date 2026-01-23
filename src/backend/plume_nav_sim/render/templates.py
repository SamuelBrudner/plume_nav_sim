"""Rendering templates for RGB arrays and matplotlib visualization."""

from __future__ import annotations

import contextlib
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from plume_nav_sim.constants import (
    BACKEND_PRIORITY_LIST,
    MATPLOTLIB_DEFAULT_FIGSIZE,
    PERFORMANCE_TARGET_HUMAN_RENDER_MS,
    PERFORMANCE_TARGET_RGB_RENDER_MS,
    RGB_DTYPE,
)
from plume_nav_sim.core.enums import RenderMode
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.types import RGBArray

from .colormaps import ColorScheme, create_default_scheme

DEFAULT_RGB_TEMPLATE_CONFIG = {
    "colormap": "gray",
    "agent_marker_size": (3, 3),
    "source_marker_size": (5, 5),
    "performance_optimized": True,
    "caching_enabled": True,
}

DEFAULT_MATPLOTLIB_TEMPLATE_CONFIG = {
    "figsize": (8, 8),
    "colormap": "gray",
    "backend_fallback": True,
    "interactive_updates": True,
    "animation_interval": 0.001,
}

TEMPLATE_CACHE_SIZE = 20
PERFORMANCE_MONITORING_ENABLED = True
TEMPLATE_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _validate_positive_grid_size(grid_size: GridSize) -> None:
    if grid_size.width <= 0 or grid_size.height <= 0:
        raise ValueError("Grid size dimensions must be positive")


def _optimize_scheme_for_mode(
    scheme: Optional[ColorScheme], mode: RenderMode
) -> ColorScheme:
    scheme = scheme or create_default_scheme()
    try:
        scheme.optimize_for_render_mode(mode)
    except Exception as exc:
        warnings.warn(f"Color scheme optimization failed: {exc}")
    return scheme


def _merge_options(
    defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    merged = defaults.copy()
    if overrides:
        merged.update(overrides)
    return merged


def _calculate_figsize_for_grid(grid_size: GridSize) -> Tuple[float, float]:
    base = 8.0
    aspect = grid_size.width / grid_size.height
    if aspect > 1.5:
        return (base * 1.2, base / 1.2)
    if aspect < 0.67:
        return (base / 1.2, base * 1.2)
    return (base, base)


def _backend_chain_with_preference(preferred: Optional[str]) -> list[str]:
    chain = BACKEND_PRIORITY_LIST.copy()
    if preferred:
        if preferred in chain:
            chain.remove(preferred)
        chain.insert(0, preferred)
    return chain


class TemplateQuality(Enum):
    ULTRA_FAST = "ultra_fast"
    FAST = "fast"
    STANDARD = "standard"
    QUALITY = "quality"
    PREMIUM = "premium"

    @property
    def performance_target_ms(self) -> float:
        targets = {
            "ultra_fast": 1.0,
            "fast": 3.0,
            "standard": 5.0,
            "quality": 10.0,
            "premium": float("inf"),
        }
        return targets[self.value]


@dataclass
class TemplateConfig:
    """Configuration for rendering templates."""

    grid_size: GridSize
    color_scheme: Optional[ColorScheme] = None
    quality_level: TemplateQuality = TemplateQuality.STANDARD
    marker_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "agent_size": (3, 3),
            "agent_color": [255, 0, 0],
            "source_size": (5, 5),
            "source_color": [255, 255, 255],
            "background_color": [0, 0, 0],
        }
    )
    performance_options: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_caching": True,
            "vectorized_operations": True,
            "memory_optimization": True,
        }
    )
    backend_preferences: Dict[str, Any] = field(
        default_factory=lambda: {
            "preferred_backend": "TkAgg",
            "fallback_chain": ["TkAgg", "Qt5Agg", "Agg"],
            "headless_compatible": True,
        }
    )
    caching_enabled: bool = True
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.color_scheme is None:
            self.color_scheme = create_default_scheme()
        _validate_positive_grid_size(self.grid_size)

    def validate(self, strict_mode: bool = False) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        if not (8 <= self.grid_size.width <= 2048):
            errors.append(
                f"Grid width {self.grid_size.width} outside reasonable range [8, 2048]"
            )
        if not (8 <= self.grid_size.height <= 2048):
            errors.append(
                f"Grid height {self.grid_size.height} outside reasonable range [8, 2048]"
            )

        if self.color_scheme is not None:
            try:
                self.color_scheme.get_concentration_colormap()
            except Exception as exc:
                errors.append(f"Color scheme validation failed: {exc}")

        agent_size = self.marker_config.get("agent_size", (3, 3))
        if not isinstance(agent_size, (tuple, list)) or len(agent_size) != 2:
            errors.append("Agent marker size must be (width, height) tuple")

        if (
            self.performance_options.get("enable_caching", True)
            and not self.caching_enabled
        ):
            errors.append("Performance caching enabled but template caching disabled")

        if strict_mode:
            memory_estimate = (self.grid_size.width * self.grid_size.height * 4) / (
                1024 * 1024
            )
            if memory_estimate > 100:
                errors.append(
                    f"Estimated memory usage {memory_estimate:.1f}MB exceeds strict limit"
                )
            preferred_backend = self.backend_preferences.get("preferred_backend")
            if preferred_backend:
                with contextlib.suppress(Exception):
                    import matplotlib

                    available = matplotlib.backend_bases.Backend._backend_map.keys()
                    if preferred_backend not in available:
                        errors.append(
                            f"Preferred backend '{preferred_backend}' not available"
                        )

        return (len(errors) == 0), errors


class BaseRenderTemplate(ABC):
    """Base class for render templates with simple performance tracking."""

    def __init__(self, config: TemplateConfig):
        is_valid, errors = config.validate()
        if not is_valid:
            raise ValueError(f"Invalid template configuration: {errors}")

        self.config = config
        self.color_scheme = config.color_scheme or create_default_scheme()
        self._initialized = False
        self._render_cache: Optional[Dict[Any, Any]] = (
            {} if config.caching_enabled else None
        )
        self._render_times: List[float] = []
        self._last_render_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0

    def initialize(self) -> bool:
        try:
            is_valid, errors = self.config.validate(strict_mode=True)
            if not is_valid:
                raise RuntimeError(
                    f"Template configuration validation failed: {errors}"
                )

            self._initialize_template_resources()

            test_concentration = np.zeros(
                (self.config.grid_size.height, self.config.grid_size.width),
                dtype=np.float32,
            )
            test_agent_pos = Coordinates(
                self.config.grid_size.width // 2,
                self.config.grid_size.height // 2,
            )
            test_source_pos = Coordinates(
                max(self.config.grid_size.width // 4, 0),
                max(self.config.grid_size.height // 4, 0),
            )
            self._execute_render(test_concentration, test_agent_pos, test_source_pos)

            self._initialized = True
            return True
        except Exception as exc:
            warnings.warn(f"Template initialization failed: {exc}")
            self._initialized = False
            return False

    @abstractmethod
    def _initialize_template_resources(self) -> None:
        pass

    @abstractmethod
    def _cleanup_template_resources(self) -> None:
        pass

    @abstractmethod
    def _execute_render(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
    ) -> Optional[RGBArray]:
        pass

    def _cache_key(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
    ) -> Optional[Tuple[Any, ...]]:
        return None

    def render(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
    ) -> Optional[RGBArray]:
        if not self._initialized:
            raise RuntimeError("Template not initialized - call initialize() first")

        cache_key = None
        if self._render_cache is not None:
            cache_key = self._cache_key(
                concentration_field, agent_position, source_position
            )
            if cache_key is not None and cache_key in self._render_cache:
                self._cache_hits += 1
                cached = self._render_cache[cache_key]
                if isinstance(cached, np.ndarray):
                    return cached.copy()
                return cached

        start = time.perf_counter()
        result = self._execute_render(
            concentration_field, agent_position, source_position
        )
        duration = time.perf_counter() - start

        if PERFORMANCE_MONITORING_ENABLED:
            self._render_times.append(duration)
            self._last_render_time = duration

        if self._render_cache is not None and cache_key is not None:
            if len(self._render_cache) >= TEMPLATE_CACHE_SIZE:
                oldest = next(iter(self._render_cache))
                self._render_cache.pop(oldest, None)
            self._cache_misses += 1
            if isinstance(result, np.ndarray):
                self._render_cache[cache_key] = result.copy()
            else:
                self._render_cache[cache_key] = result

        return result

    def cleanup(self) -> None:
        self._cleanup_template_resources()
        if self._render_cache is not None:
            self._render_cache.clear()
        self._initialized = False

    def get_performance_metrics(self) -> Dict[str, Any]:
        avg = float(np.mean(self._render_times)) if self._render_times else 0.0
        return {
            "total_renders": len(self._render_times),
            "average_render_time": avg,
            "last_render_time": self._last_render_time,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }


class RGBTemplate(BaseRenderTemplate):
    """RGB array template for programmatic rendering."""

    def __init__(self, config: TemplateConfig):
        super().__init__(config)
        self._color_lookup_table: Optional[np.ndarray] = None
        self._agent_marker_pattern: Optional[np.ndarray] = None
        self._source_marker_pattern: Optional[np.ndarray] = None
        self._optimization_enabled = False

    def _initialize_template_resources(self) -> None:
        agent_size = self.config.marker_config.get("agent_size", (3, 3))
        source_size = self.config.marker_config.get("source_size", (5, 5))
        agent_color = self.config.marker_config.get("agent_color", [255, 0, 0])
        source_color = self.config.marker_config.get("source_color", [255, 255, 255])

        self._agent_marker_pattern = self._build_marker_pattern(
            agent_size, agent_color, cross=False
        )
        self._source_marker_pattern = self._build_marker_pattern(
            source_size, source_color, cross=True
        )
        self._optimization_enabled = True

    def _cleanup_template_resources(self) -> None:
        self._color_lookup_table = None
        self._agent_marker_pattern = None
        self._source_marker_pattern = None
        self._optimization_enabled = False

    def _cache_key(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
    ) -> Optional[Tuple[Any, ...]]:
        try:
            data_ptr = concentration_field.__array_interface__["data"][0]
        except Exception:
            data_ptr = id(concentration_field)
        return (
            data_ptr,
            concentration_field.shape,
            agent_position.to_tuple(),
            source_position.to_tuple(),
        )

    def _build_marker_pattern(
        self, size: Tuple[int, int], color: List[int], cross: bool
    ) -> np.ndarray:
        height = max(int(size[0]), 1)
        width = max(int(size[1]), 1)
        pattern = np.zeros((height, width, 3), dtype=RGB_DTYPE)
        color_arr = np.array(color, dtype=RGB_DTYPE)
        if cross:
            cy = height // 2
            cx = width // 2
            pattern[cy, :, :] = color_arr
            pattern[:, cx, :] = color_arr
        else:
            pattern[:, :, :] = color_arr
        return pattern

    def _apply_marker(
        self, rgb_array: np.ndarray, position: Coordinates, pattern: np.ndarray
    ) -> None:
        ph, pw = pattern.shape[:2]
        half_h = ph // 2
        half_w = pw // 2

        y0 = position.y - half_h
        x0 = position.x - half_w
        y1 = y0 + ph
        x1 = x0 + pw

        y_start = max(0, y0)
        x_start = max(0, x0)
        y_end = min(rgb_array.shape[0], y1)
        x_end = min(rgb_array.shape[1], x1)

        if y_start >= y_end or x_start >= x_end:
            return

        pattern_y0 = y_start - y0
        pattern_x0 = x_start - x0
        pattern_y1 = pattern_y0 + (y_end - y_start)
        pattern_x1 = pattern_x0 + (x_end - x_start)

        rgb_slice = rgb_array[y_start:y_end, x_start:x_end]
        pattern_slice = pattern[pattern_y0:pattern_y1, pattern_x0:pattern_x1]

        mask = np.any(pattern_slice > 0, axis=2)
        rgb_slice[mask] = pattern_slice[mask]

    def _execute_render(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
    ) -> RGBArray:
        if concentration_field.shape != (
            self.config.grid_size.height,
            self.config.grid_size.width,
        ):
            raise ValueError(
                f"Concentration field shape {concentration_field.shape} does not match "
                f"grid {self.config.grid_size.width}x{self.config.grid_size.height}"
            )

        field = np.asarray(concentration_field, dtype=np.float32)
        field = np.clip(field, 0.0, 1.0)
        gray = (field * 255).astype(RGB_DTYPE)
        rgb_array = np.repeat(gray[:, :, None], 3, axis=2)

        background = self.config.marker_config.get("background_color", [0, 0, 0])
        background_rgb = np.array(background, dtype=RGB_DTYPE)
        zero_mask = field <= 0.0
        if np.any(zero_mask):
            rgb_array[zero_mask] = background_rgb

        if self._agent_marker_pattern is not None:
            self._apply_marker(rgb_array, agent_position, self._agent_marker_pattern)
        if self._source_marker_pattern is not None:
            self._apply_marker(rgb_array, source_position, self._source_marker_pattern)

        return rgb_array

    def _rgb_collect_test_results(
        self,
        test_scenarios: Dict[str, Any],
        rgb_target_ms: float,
        analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for scenario_name, scenario_config in test_scenarios.items():
            grid_sizes = scenario_config.get("grid_sizes", [(64, 64), (128, 128)])
            for grid in grid_sizes:
                result = self._run_single_rgb_scenario(
                    scenario_name, grid, rgb_target_ms
                )
                results.append(result)

                if result.get("success") and not result.get("meets_target"):
                    analysis["meets_target"] = False
                    analysis["target_violations"].append(
                        {
                            "scenario": result["scenario"],
                            "actual_time_ms": result["avg_render_time_ms"],
                            "target_time_ms": rgb_target_ms,
                            "excess_time_ms": result["avg_render_time_ms"]
                            - rgb_target_ms,
                        }
                    )
        return results

    def _run_single_rgb_scenario(
        self, scenario_name: str, grid: Tuple[int, int], rgb_target_ms: float
    ) -> Dict[str, Any]:
        test_field = np.random.rand(*grid).astype(np.float32)
        test_agent = Coordinates(grid[1] // 2, grid[0] // 2)
        test_source = Coordinates(grid[1] // 4, grid[0] // 4)

        timings: List[float] = []
        for _ in range(10):
            start = time.perf_counter()
            try:
                result = self.render(test_field, test_agent, test_source)
                elapsed = time.perf_counter() - start
                timings.append(elapsed)

                if not isinstance(result, np.ndarray):
                    raise ValueError("RGB template must return numpy array")
                if result.shape != (*grid, 3):
                    raise ValueError(f"Invalid RGB array shape: {result.shape}")
                if result.dtype != RGB_DTYPE:
                    raise ValueError(f"Invalid RGB array dtype: {result.dtype}")
            except Exception as exc:
                return {
                    "scenario": f"{scenario_name}_{grid[0]}x{grid[1]}",
                    "success": False,
                    "error": str(exc),
                }

        if timings:
            avg = np.mean(timings)
            mx = np.max(timings)
            meets = avg <= (rgb_target_ms / 1000.0)
            return {
                "scenario": f"{scenario_name}_{grid[0]}x{grid[1]}",
                "success": True,
                "avg_render_time_ms": avg * 1000,
                "max_render_time_ms": mx * 1000,
                "meets_target": meets,
            }

        return {
            "scenario": f"{scenario_name}_{grid[0]}x{grid[1]}",
            "success": False,
            "error": "No timings collected",
        }


class MatplotlibTemplate(BaseRenderTemplate):
    """Matplotlib template for human-mode visualization."""

    def __init__(self, config: TemplateConfig):
        super().__init__(config)
        self.figure = None
        self.axes = None
        self.image = None
        self._agent_scatter = None
        self._source_scatter = None
        self.active_backend = plt.get_backend()

    def configure_backend(self, backend: str) -> bool:
        try:
            plt.switch_backend(backend)
            self.active_backend = backend
            return True
        except Exception:
            return False

    def _initialize_template_resources(self) -> None:
        preferred_backend = self.config.backend_preferences.get("preferred_backend")
        if preferred_backend:
            self.configure_backend(preferred_backend)

        figsize = self.config.backend_preferences.get(
            "figsize", MATPLOTLIB_DEFAULT_FIGSIZE
        )
        self.figure, self.axes = plt.subplots(figsize=figsize)
        self.axes.set_axis_off()
        self.image = self.axes.imshow(
            np.zeros(
                (self.config.grid_size.height, self.config.grid_size.width),
                dtype=np.float32,
            ),
            cmap=self.config.custom_parameters.get("colormap", "gray"),
            vmin=0.0,
            vmax=1.0,
        )
        agent_color = (
            np.array(self.config.marker_config.get("agent_color", [255, 0, 0])) / 255.0
        )
        source_color = (
            np.array(self.config.marker_config.get("source_color", [255, 255, 255]))
            / 255.0
        )
        self._agent_scatter = self.axes.scatter([], [], c=[agent_color], s=20)
        self._source_scatter = self.axes.scatter([], [], c=[source_color], s=30)

    def _cleanup_template_resources(self) -> None:
        if self.figure is not None:
            plt.close(self.figure)
        self.figure = None
        self.axes = None
        self.image = None
        self._agent_scatter = None
        self._source_scatter = None

    def _execute_render(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
    ) -> None:
        if self.image is None or self.axes is None:
            raise RuntimeError("Matplotlib template not initialized")

        self.image.set_data(concentration_field)
        if self._agent_scatter is not None:
            self._agent_scatter.set_offsets(
                np.array([[agent_position.x, agent_position.y]])
            )
        if self._source_scatter is not None:
            self._source_scatter.set_offsets(
                np.array([[source_position.x, source_position.y]])
            )

        if self.figure is not None:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
        return None


def create_rgb_template(
    grid_size: GridSize,
    color_scheme: Optional[ColorScheme] = None,
    quality_level: Optional[TemplateQuality] = None,
    template_options: Optional[Dict[str, Any]] = None,
) -> RGBTemplate:
    _validate_positive_grid_size(grid_size)

    color_scheme = _optimize_scheme_for_mode(color_scheme, RenderMode.RGB_ARRAY)
    options = _merge_options(DEFAULT_RGB_TEMPLATE_CONFIG, template_options)

    marker_config = {
        "agent_size": options.get("agent_marker_size", (3, 3)),
        "agent_color": list(options.get("agent_color", color_scheme.agent_color)),
        "source_size": options.get("source_marker_size", (5, 5)),
        "source_color": list(options.get("source_color", color_scheme.source_color)),
        "background_color": list(color_scheme.background_color),
    }

    config = TemplateConfig(
        grid_size=grid_size,
        color_scheme=color_scheme,
        quality_level=quality_level or TemplateQuality.STANDARD,
        marker_config=marker_config,
        performance_options={
            "enable_caching": options.get("caching_enabled", True),
            "vectorized_operations": options.get("performance_optimized", True),
        },
        caching_enabled=options.get("caching_enabled", True),
        custom_parameters={"colormap": options.get("colormap", "gray")},
    )

    template = RGBTemplate(config)
    if not template.initialize():
        raise RuntimeError("RGB template initialization failed")
    return template


def create_matplotlib_template(
    grid_size: GridSize,
    color_scheme: Optional[ColorScheme] = None,
    preferred_backend: Optional[str] = None,
    template_options: Optional[Dict[str, Any]] = None,
) -> MatplotlibTemplate:
    _validate_positive_grid_size(grid_size)

    color_scheme = _optimize_scheme_for_mode(color_scheme, RenderMode.HUMAN)
    options = _merge_options(DEFAULT_MATPLOTLIB_TEMPLATE_CONFIG, template_options)

    backend_chain = _backend_chain_with_preference(preferred_backend)
    marker_config = {
        "agent_size": options.get("agent_marker_size", (3, 3)),
        "agent_color": list(options.get("agent_color", color_scheme.agent_color)),
        "source_size": options.get("source_marker_size", (5, 5)),
        "source_color": list(options.get("source_color", color_scheme.source_color)),
        "background_color": list(color_scheme.background_color),
    }

    config = TemplateConfig(
        grid_size=grid_size,
        color_scheme=color_scheme,
        quality_level=TemplateQuality.STANDARD,
        marker_config=marker_config,
        backend_preferences={
            "preferred_backend": backend_chain[0] if backend_chain else None,
            "fallback_chain": backend_chain,
            "headless_compatible": True,
            "figsize": options.get("figsize", _calculate_figsize_for_grid(grid_size)),
        },
        caching_enabled=False,
        custom_parameters={"colormap": options.get("colormap", "gray")},
    )

    template = MatplotlibTemplate(config)
    if not template.initialize():
        raise RuntimeError("Matplotlib template initialization failed")
    return template


def create_custom_template(
    config: TemplateConfig,
    template_type: str,
    template_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    allow_override: bool = False,
) -> Union[RGBTemplate, MatplotlibTemplate]:
    ok, errors = config.validate(strict_mode=True)
    if not ok:
        raise ValueError(f"Invalid template configuration: {errors}")

    normalized = template_type.lower()
    if normalized in {"rgb", "rgb_array", "array"}:
        template = RGBTemplate(config)
    elif normalized in {"matplotlib", "human", "mpl"}:
        template = MatplotlibTemplate(config)
    else:
        raise ValueError(f"Template type '{template_type}' not supported")
    if template_name:
        register_template(template_name, template, metadata, allow_override)
    return template


def register_template(
    template_name: str,
    template: Union[RGBTemplate, MatplotlibTemplate],
    metadata: Optional[Dict[str, Any]] = None,
    allow_override: bool = False,
) -> None:
    _validate_template_name(template_name)
    if template_name in TEMPLATE_REGISTRY and not allow_override:
        raise ValueError(f"Template '{template_name}' already registered.")
    TEMPLATE_REGISTRY[template_name] = {
        "template": template,
        "metadata": metadata or {},
    }


def _validate_template_name(template_name: str) -> None:
    if not isinstance(template_name, str) or len(template_name.strip()) < 3:
        raise ValueError("Template name must be a string with at least 3 characters")
    if not all(ch.isalnum() or ch in "-_" for ch in template_name):
        raise ValueError(
            "Template name must contain only alphanumeric characters, hyphens, and underscores"
        )


def get_template_registry(
    template_type_filter: Optional[str] = None,
    include_metadata: bool = False,
    include_templates: bool = False,
) -> Dict[str, Any]:
    if not TEMPLATE_REGISTRY:
        return {"templates": {}, "stats": {"total_templates": 0}}

    filtered: Dict[str, Dict[str, Any]] = {}
    for name, info in TEMPLATE_REGISTRY.items():
        if template_type_filter:
            if template_type_filter == "rgb" and not isinstance(
                info["template"], RGBTemplate
            ):
                continue
            if template_type_filter == "matplotlib" and not isinstance(
                info["template"], MatplotlibTemplate
            ):
                continue

        entry = {"template_type": type(info["template"]).__name__}
        if include_metadata:
            entry["metadata"] = info["metadata"]
        if include_templates:
            entry["template"] = info["template"]
        filtered[name] = entry

    return {"templates": filtered, "stats": {"total_templates": len(filtered)}}


def optimize_template(
    template: Union[RGBTemplate, MatplotlibTemplate],
    optimization_targets: Dict[str, Any],
    usage_statistics: Optional[Dict[str, Any]] = None,
) -> Tuple[Union[RGBTemplate, MatplotlibTemplate], Dict[str, Any]]:
    current_metrics = template.get_performance_metrics()
    report = {
        "original_performance": current_metrics.copy(),
        "optimizations_applied": [],
        "recommendations": [],
    }

    target_render_time_ms = optimization_targets.get(
        "target_render_time_ms", PERFORMANCE_TARGET_RGB_RENDER_MS
    )
    current_rt_ms = current_metrics.get("average_render_time", 0.0) * 1000
    if current_rt_ms > target_render_time_ms:
        report["recommendations"].append(
            "Reduce quality level or grid size to hit target latency"
        )

    if usage_statistics and usage_statistics.get("repeated_renders", 0.0) > 0.3:
        if not template.config.caching_enabled:
            report["recommendations"].append("Enable caching for repeated renders")

    if not report["recommendations"]:
        report["recommendations"].append(
            "Template already well-optimized for current targets"
        )

    return template, report


def validate_template_performance(
    template: Union[RGBTemplate, MatplotlibTemplate],
    test_scenarios: Dict[str, Any],
    strict_validation: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    report = {
        "validation_passed": True,
        "template_info": {
            "type": type(template).__name__,
            "quality_level": template.config.quality_level.value,
            "grid_size": f"{template.config.grid_size.width}x{template.config.grid_size.height}",
            "caching_enabled": template.config.caching_enabled,
        },
        "test_results": {},
        "performance_analysis": {},
        "compatibility_results": {},
        "recommendations": [],
    }

    target_ms, template_kind = _determine_targets_for_template(template)
    report["performance_analysis"]["target_time_ms"] = target_ms

    all_results = _run_benchmarks_for_template(
        template, test_scenarios, target_ms, strict_validation, report
    )

    if all_results:
        successful = [r for r in all_results if r.get("success_rate", 0) > 0.8]
        if successful:
            average_times = [r["average_time_ms"] for r in successful]
            report["performance_analysis"].update(
                {
                    "overall_average_time_ms": np.mean(average_times),
                    "overall_max_time_ms": np.max(
                        [r["max_time_ms"] for r in successful]
                    ),
                    "overall_success_rate": np.mean(
                        [r["success_rate"] for r in successful]
                    ),
                    "target_compliance_rate": np.mean(
                        [r["meets_target"] for r in successful]
                    ),
                }
            )
            if np.mean(average_times) > 0:
                report["performance_analysis"]["performance_consistency"] = {
                    "std_deviation_ms": float(np.std(average_times)),
                    "coefficient_of_variation": float(
                        np.std(average_times) / np.mean(average_times)
                    ),
                }

    if isinstance(template, MatplotlibTemplate):
        report["compatibility_results"] = {}
    if strict_validation:
        if report["performance_analysis"].get("overall_success_rate", 1.0) < 1.0:
            report["validation_passed"] = False
            report["strict_validation_failures"] = ["Not all test iterations succeeded"]

    if not report["validation_passed"]:
        report["recommendations"].extend(
            [
                f"Template does not meet {template_kind} performance targets (<{target_ms}ms)",
                "Consider reducing quality level for better performance",
                "Enable caching if not already enabled for repetitive operations",
            ]
        )

    avg_time = report["performance_analysis"].get("overall_average_time_ms", 0)
    if avg_time > target_ms * 1.5:
        report["recommendations"].append(
            "Significant performance optimization needed - consider ULTRA_FAST quality level"
        )
    elif avg_time > target_ms * 1.2:
        report["recommendations"].append("Minor performance optimization recommended")

    return report["validation_passed"], report


def _determine_targets_for_template(
    template: Union[RGBTemplate, MatplotlibTemplate]
) -> Tuple[float, str]:
    if isinstance(template, RGBTemplate):
        return PERFORMANCE_TARGET_RGB_RENDER_MS, "RGB"
    return PERFORMANCE_TARGET_HUMAN_RENDER_MS, "Matplotlib"


def _run_benchmarks_for_template(
    template: Union[RGBTemplate, MatplotlibTemplate],
    test_scenarios: Dict[str, Any],
    target_ms: float,
    strict_validation: bool,
    report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    for scenario_name, scenario_config in test_scenarios.items():
        scenario_results = {"scenario_name": scenario_name, "tests": [], "summary": {}}
        grid_sizes = scenario_config.get("grid_sizes", [(64, 64), (128, 128)])
        iterations = scenario_config.get(
            "iterations", 10 if isinstance(template, RGBTemplate) else 5
        )

        for grid in grid_sizes:
            tr = _run_single_benchmark_scenario(
                template,
                scenario_name,
                grid,
                iterations,
                target_ms,
                strict_validation,
                report,
            )
            if tr:
                scenario_results["tests"].append(tr)
                all_results.append(tr)
            else:
                scenario_results["tests"].append(
                    {
                        "grid_size": f"{grid[0]}x{grid[1]}",
                        "success_rate": 0.0,
                        "error": "All iterations failed",
                    }
                )

        successful_tests = [
            t for t in scenario_results["tests"] if t.get("success_rate", 0) > 0.5
        ]
        if successful_tests:
            scenario_results["summary"] = {
                "overall_success_rate": np.mean(
                    [t["success_rate"] for t in successful_tests]
                ),
                "average_time_ms": np.mean(
                    [t["average_time_ms"] for t in successful_tests]
                ),
                "target_compliance_rate": np.mean(
                    [t["meets_target"] for t in successful_tests]
                ),
            }

        report["test_results"][scenario_name] = scenario_results

    return all_results


def _run_single_benchmark_scenario(
    template: Union[RGBTemplate, MatplotlibTemplate],
    scenario_name: str,
    grid: Tuple[int, int],
    iterations: int,
    target_ms: float,
    strict_validation: bool,
    report: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    test_field = np.random.rand(*grid).astype(np.float32)
    test_agent = Coordinates(grid[1] // 2, grid[0] // 2)
    test_source = Coordinates(grid[1] // 4, grid[0] // 4)

    times: List[float] = []
    success = 0

    for iteration in range(iterations):
        try:
            elapsed, _ = _execute_benchmark_iteration(
                template, test_field, test_agent, test_source, grid
            )
            times.append(elapsed)
            success += 1
        except Exception as exc:
            report["validation_passed"] = False
            times.append(-1)
            if strict_validation:
                report["test_results"][f"{scenario_name}_{grid[0]}x{grid[1]}"] = {
                    "success": False,
                    "error": str(exc),
                    "iteration": iteration,
                }
                break

    successful_times = [t for t in times if t > 0]
    if not successful_times:
        report["validation_passed"] = False
        return None

    tr = {
        "grid_size": f"{grid[0]}x{grid[1]}",
        "success_rate": success / iterations,
        "average_time_ms": np.mean(successful_times) * 1000,
        "max_time_ms": np.max(successful_times) * 1000,
        "min_time_ms": np.min(successful_times) * 1000,
        "std_time_ms": np.std(successful_times) * 1000,
        "meets_target": np.mean(successful_times) * 1000 <= target_ms,
        "iterations": iterations,
    }
    if not tr["meets_target"]:
        report["validation_passed"] = False
    return tr


def _execute_benchmark_iteration(
    template: Union[RGBTemplate, MatplotlibTemplate],
    test_field: np.ndarray,
    test_agent: Coordinates,
    test_source: Coordinates,
    grid: Tuple[int, int],
) -> Tuple[float, Optional[float]]:
    start = time.perf_counter()
    result = template.render(test_field, test_agent, test_source)
    elapsed = time.perf_counter() - start

    if isinstance(template, RGBTemplate):
        if not isinstance(result, np.ndarray):
            raise ValueError("RGB template must return numpy array")
        if result.shape != (*grid, 3):
            raise ValueError(f"Invalid RGB shape: {result.shape}")
        if result.dtype != RGB_DTYPE:
            raise ValueError(f"Invalid RGB dtype: {result.dtype}")

    return elapsed, None


__all__ = [
    "BaseRenderTemplate",
    "RGBTemplate",
    "MatplotlibTemplate",
    "TemplateConfig",
    "TemplateQuality",
    "create_rgb_template",
    "create_matplotlib_template",
    "create_custom_template",
    "register_template",
    "get_template_registry",
    "optimize_template",
    "validate_template_performance",
]
