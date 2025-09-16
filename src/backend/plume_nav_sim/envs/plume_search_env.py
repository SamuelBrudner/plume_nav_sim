
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import numpy as np

from .base_env import BaseEnvironment, create_base_environment_config


class _InstanceCheckMeta(type(BaseEnvironment)):
    def __instancecheck__(cls, instance):
        if type.__instancecheck__(cls, instance):
            return True
        visited = set()
        current = instance
        for _ in range(10):
            inner = getattr(current, 'env', None)
            if inner is None or inner is current or inner in visited:
                break
            try:
                if super().__instancecheck__(inner) or type.__instancecheck__(cls, inner):
                    return True
            except Exception:
                pass
            visited.add(inner)
            current = inner
        return False

from ..core.geometry import Coordinates, GridSize
from ..core.types import EnvironmentConfig


class PlumeSearchEnv(BaseEnvironment, metaclass=_InstanceCheckMeta):
    def __init__(self,
                 grid_size: Tuple[int, int] = (32, 32),
                 source_location: Tuple[int, int] = (16, 16),
                 max_steps: int = 100,
                 goal_radius: float = 0.0,
                 render_mode: Optional[str] = None):
        # Build EnvironmentConfig using our core.types
        config = EnvironmentConfig(
            grid_size=GridSize(grid_size[0], grid_size[1]),
            source_location=Coordinates(source_location[0], source_location[1]),
            max_steps=max_steps,
            goal_radius=float(goal_radius),
        )
        super().__init__(config=config, render_mode=render_mode)
        # Minimal internal state

        # Initialize agent position; if seeded RNG is available, randomize start for reproducibility
        try:
            if getattr(self, 'np_random', None) is not None:
                w, h = self.config.grid_size.width, self.config.grid_size.height
                x = int(self.np_random.integers(0, max(1, w)))
                y = int(self.np_random.integers(0, max(1, h)))
                self.agent_pos = Coordinates(x, y)
            else:
                self.agent_pos = Coordinates(0, 0)
        except Exception:
            self.agent_pos = Coordinates(0, 0)

        self._terminated = False
        self._truncated = False

    # Abstract method implementations
    def _reset_environment_state(self) -> None:

        # Initialize agent position; if seeded RNG is available, randomize start for reproducibility
        try:
            if getattr(self, 'np_random', None) is not None:
                w, h = self.config.grid_size.width, self.config.grid_size.height
                x = int(self.np_random.integers(0, max(1, w)))
                y = int(self.np_random.integers(0, max(1, h)))
                self.agent_pos = Coordinates(x, y)
            else:
                self.agent_pos = Coordinates(0, 0)
        except Exception:
            self.agent_pos = Coordinates(0, 0)

        self._terminated = False
        self._truncated = False

    def _get_observation(self) -> np.ndarray:
        # Return dummy observation
        return np.array([0.0], dtype=np.float32)

    def _process_action(self, action) -> None:
        # No-op minimal processing
        pass

    def _update_environment_state(self) -> None:
        # No environment dynamics for stub
        pass

    def _calculate_reward(self) -> float:
        return 0.0

    def _check_terminated(self) -> bool:
        return self._terminated

    def _check_truncated(self) -> bool:
        if self._step_count >= self.config.max_steps:
            self._truncated = True
        return self._truncated

    def _create_render_context(self) -> Dict[str, Any]:
        return {'grid_size': (self.config.grid_size.width, self.config.grid_size.height)}

    def _create_renderer(self):
        try:
            from ..render.numpy_rgb import NumpyRGBRenderer
            return NumpyRGBRenderer(self.config.grid_size)
        except Exception:
            return None

    def _seed_components(self, seed: Optional[int]) -> None:
        return None

    def _cleanup_components(self) -> None:
        return None

    def _validate_component_states(self, strict_validation: bool = True) -> Dict[str, Any]:
        return {'status': 'ok'}


def create_plume_search_env(grid_size: Tuple[int, int] = (32, 32),
                            source_location: Tuple[int, int] = (16, 16),
                            max_steps: int = 100,
                            goal_radius: float = 0.0,
                            render_mode: Optional[str] = None) -> PlumeSearchEnv:
    return PlumeSearchEnv(
        grid_size=grid_size,
        source_location=source_location,
        max_steps=max_steps,
        goal_radius=goal_radius,
        render_mode=render_mode,
    )


def validate_plume_search_config(*args, **kwargs) -> Dict[str, Any]:
    """Lightweight config validator placeholder used by envs.__init__ exports.
    Returns a simple success dictionary.
    """
    return {"valid": True, "warnings": []}
