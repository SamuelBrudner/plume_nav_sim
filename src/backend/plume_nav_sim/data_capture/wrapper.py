from __future__ import annotations

import time
from typing import Optional

import gymnasium as gym

from ..core.types import EnvironmentConfig
from .recorder import RunRecorder
from .schemas import EpisodeRecord, Position, RunMeta, StepRecord


class DataCaptureWrapper(gym.Wrapper):
    """Gymnasium wrapper that records step/episode data using RunRecorder.

    Usage:
        env = make_env(...)
        recorder = RunRecorder("results")
        wrapped = DataCaptureWrapper(env, recorder, env_config)
    """

    def __init__(
        self,
        env: gym.Env,
        recorder: RunRecorder,
        env_config: EnvironmentConfig,
        *,
        meta_overrides: Optional[dict] = None,
    ):
        super().__init__(env)
        self.recorder = recorder
        self.env_config = env_config
        # Write run meta on construction
        meta = RunMeta(
            run_id=recorder.run_id,
            experiment=recorder.experiment,
            package_version=None,
            git_sha=None,
            start_time=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),  # will be parsed by pydantic
            env_config=env_config.to_dict(),
            base_seed=None,
            episode_seeds=None,
            system=RunMeta.SystemInfo(),
        )
        if meta_overrides:
            try:
                meta = meta.model_copy(update={**meta_overrides})
            except Exception:
                # Best-effort: ignore invalid overrides
                pass
        # pydantic will coerce start_time to datetime if string in ISO format
        self.recorder.write_run_meta(meta)
        self._episode_id: Optional[str] = None
        self._episode_start_ms: Optional[float] = None
        self._episode_steps: int = 0
        self._episode_reward: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Start new episode
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._episode_start_ms = time.time() * 1000.0
        eid = info.get("episode", 0)
        self._episode_id = f"ep-{int(eid):06d}"
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_steps += 1
        self._episode_reward += float(reward)
        pos = info.get("agent_position") or info.get("agent_xy") or (0, 0)
        dist = float(info.get("distance_to_goal", 0.0))
        seed = info.get("seed")
        # Compose step record
        rec = StepRecord(
            ts=time.time(),
            run_id=self.recorder.run_id,
            episode_id=self._episode_id or "ep-unknown",
            step=self._episode_steps,
            action=int(action),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            agent_position=Position(x=int(pos[0]), y=int(pos[1])),
            distance_to_goal=dist,
            observation_summary=(
                [float(obs[0])] if hasattr(obs, "__len__") and len(obs) > 0 else None
            ),
            seed=int(seed) if isinstance(seed, (int,)) else None,
        )
        self.recorder.append_step(rec)

        if terminated or truncated:
            # finalize episode record
            duration_ms = None
            if self._episode_start_ms is not None:
                duration_ms = (time.time() * 1000.0) - self._episode_start_ms
            ep = EpisodeRecord(
                run_id=self.recorder.run_id,
                episode_id=self._episode_id or "ep-unknown",
                terminated=bool(terminated),
                truncated=bool(truncated),
                total_steps=self._episode_steps,
                total_reward=self._episode_reward,
                final_position=Position(x=int(pos[0]), y=int(pos[1])),
                final_distance_to_goal=dist,
                duration_ms=duration_ms,
                avg_step_time_ms=(
                    (duration_ms / max(1, self._episode_steps)) if duration_ms else None
                ),
            )
            self.recorder.append_episode(ep)

        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        """Passthrough to underlying env.render to preserve frames.

        Ensures that calls like env.render() or env.render("rgb_array")
        return the ndarray produced by the wrapped environment.
        """
        return self.env.render(*args, **kwargs)
