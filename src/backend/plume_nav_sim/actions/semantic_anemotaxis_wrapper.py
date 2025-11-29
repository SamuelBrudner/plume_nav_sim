from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from plume_nav_sim.core.constants import MOVEMENT_VECTORS
from plume_nav_sim.core.enums import Action
from plume_nav_sim.core.geometry import Coordinates


class SemanticAnemotaxisActionWrapper(gym.ActionWrapper):
    """Map SURGE/CAST intents to cardinal primitives using instantaneous wind.

    SURGE selects the primitive action aligned with the upwind direction
    (opposite the current wind vector). CAST samples uniformly from the
    remaining three primitives. One semantic action always maps to a single
    primitive action (no temporal macro locking).
    """

    SURGE = 0
    CAST = 1

    def __init__(
        self, env: gym.Env, *, rng: Optional[np.random.Generator] = None
    ) -> None:
        super().__init__(env)

        self._primitive_actions = np.array(
            [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT], dtype=np.int64
        )
        self._movement_vectors = np.array(
            [MOVEMENT_VECTORS[int(a)] for a in self._primitive_actions],
            dtype=np.float32,
        )

        if not isinstance(env.action_space, gym.spaces.Discrete) or (
            env.action_space.n != len(self._primitive_actions)
        ):
            raise ValueError(
                "SemanticAnemotaxisActionWrapper requires a Discrete(4) "
                "primitive action space (UP, RIGHT, DOWN, LEFT)."
            )

        self.action_space = gym.spaces.Discrete(2)
        self._rng = rng or getattr(env, "_rng", None) or np.random.default_rng()

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._sync_rng()
        return obs, info

    def action(self, action: Any) -> int:  # type: ignore[override]
        semantic_action = self._coerce_semantic_action(action)
        upwind_action = self._compute_upwind_action()

        if semantic_action == self.SURGE:
            return upwind_action

        return self._sample_cast_action(exclude_action=upwind_action)

    def _compute_upwind_action(self) -> int:
        wind_vector = self._get_wind_vector()
        upwind_vector = -wind_vector

        if not np.any(np.isfinite(upwind_vector)) or upwind_vector.shape != (2,):
            return int(Action.UP)

        magnitude = float(np.linalg.norm(upwind_vector))
        if magnitude < 1e-8:
            return int(Action.UP)

        scores = self._movement_vectors @ upwind_vector
        best_idx = int(np.argmax(scores))
        return int(self._primitive_actions[best_idx])

    def _sample_cast_action(self, *, exclude_action: int) -> int:
        candidates = [
            int(a) for a in self._primitive_actions if int(a) != int(exclude_action)
        ]
        if not candidates:
            return int(exclude_action)

        rng = self._rng or np.random.default_rng()
        choice_idx = int(rng.integers(0, len(candidates)))
        return candidates[choice_idx]

    def _get_wind_vector(self) -> np.ndarray:
        env_state = self._try_build_env_state()

        wind_field = None
        agent_state = None
        if isinstance(env_state, dict):
            wind_field = env_state.get("wind_field")
            agent_state = env_state.get("agent_state")

        wind_field = wind_field or getattr(self.env, "_wind_field", None)
        if wind_field is None:
            wind_field = getattr(self.env, "wind_field", None)

        agent_state = agent_state or getattr(self.env, "_agent_state", None)
        if agent_state is None:
            agent_state = getattr(self.env, "agent_state", None)

        position = getattr(agent_state, "position", None) or Coordinates(0, 0)

        if wind_field is None:
            return np.zeros(2, dtype=np.float32)

        try:
            vector = wind_field.sample(position)
            arr = np.asarray(vector, dtype=np.float32).reshape(-1)
            if arr.shape == (2,) and np.all(np.isfinite(arr)):
                return arr
        except Exception:
            return np.zeros(2, dtype=np.float32)

        return np.zeros(2, dtype=np.float32)

    def _sync_rng(self) -> None:
        for attr in ("_rng", "np_random"):
            env_rng = getattr(self.env, attr, None)
            if isinstance(env_rng, np.random.Generator):
                self._rng = env_rng
                return

    def _coerce_semantic_action(self, action: Any) -> int:
        if isinstance(action, (int, np.integer)):
            semantic = int(action)
        else:
            raise ValueError(
                f"Semantic action must be int-compatible, got {type(action).__name__}"
            )

        if semantic not in (self.SURGE, self.CAST):
            raise ValueError(
                f"Semantic action must be 0 (SURGE) or 1 (CAST), got {semantic}"
            )
        return semantic

    def _try_build_env_state(self) -> Optional[Dict[str, Any]]:
        builder = getattr(self.env, "_build_env_state_dict", None)
        if callable(builder):
            try:
                return builder()
            except Exception:
                return None
        return None
