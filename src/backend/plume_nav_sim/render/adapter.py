"""RendererAdapter for lightweight render-mode capability checks."""

from __future__ import annotations

from typing import Any, Optional

from ..core.types import GridSize, RenderMode
from .matplotlib_viz import MatplotlibRenderer
from .numpy_rgb import NumpyRGBRenderer

__all__ = ["RendererAdapter"]


class RendererAdapter:
    """Expose a minimal renderer interface across RGB and matplotlib backends."""

    def __init__(self, width: int, height: int) -> None:
        grid = GridSize(width=int(width), height=int(height))
        try:
            self.rgb_renderer: Optional[NumpyRGBRenderer] = NumpyRGBRenderer(grid)
        except Exception:
            self.rgb_renderer = None
        try:
            self.matplotlib_renderer: Optional[MatplotlibRenderer] = MatplotlibRenderer(
                grid
            )
        except Exception:
            self.matplotlib_renderer = None

    def supports_render_mode(self, mode: RenderMode) -> bool:
        if mode == RenderMode.RGB_ARRAY:
            return self.rgb_renderer is not None
        if mode == RenderMode.HUMAN:
            if self.matplotlib_renderer is None:
                return False
            try:
                return bool(self.matplotlib_renderer.supports_render_mode(mode))
            except Exception:
                return False
        return False

    def configure_interactive_mode(self, cfg: Any) -> bool:
        if self.matplotlib_renderer is None:
            return False
        return bool(self.matplotlib_renderer.configure_interactive_mode(cfg))

    def enable_interactive_mode(self) -> None:
        if self.matplotlib_renderer is not None:
            self.matplotlib_renderer.enable_interactive_mode()

    def disable_interactive_mode(self) -> None:
        if self.matplotlib_renderer is not None:
            self.matplotlib_renderer.disable_interactive_mode()

    def set_interactive_mode(self, enable: bool = True, **kwargs: Any) -> bool:
        if self.matplotlib_renderer is None:
            return False
        return bool(self.matplotlib_renderer.set_interactive_mode(enable, **kwargs))
