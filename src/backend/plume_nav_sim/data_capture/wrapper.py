from __future__ import annotations

import time
from typing import Any, Optional

import gymnasium as gym

from ..envs.config_types import EnvironmentConfig
from .recorder import RunRecorder
from .schemas import EpisodeRecord, Position, RunMeta, StepRecord


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _get_package_version() -> Optional[str]:
    try:
        from plume_nav_sim import __version__

        return __version__
    except (ImportError, AttributeError):
        return None


def _build_simulation_metadata(
    env_config: dict, meta_overrides: Optional[dict] = None
) -> Optional[dict]:
    """Auto-generate SimulationMetadata dict from env config for FAIR provenance."""
    try:
        import hashlib
        import json
        from datetime import datetime, timezone

        config_str = json.dumps(env_config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        seed = None
        if isinstance(meta_overrides, dict):
            seed = meta_overrides.get("base_seed")

        return {
            "software_name": "plume-nav-sim",
            "software_version": _get_package_version(),
            "config_hash": config_hash,
            "random_seed": seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return None


class DataCaptureWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        recorder: RunRecorder,
        env_config: EnvironmentConfig | dict,
        *,
        meta_overrides: Optional[dict] = None,
    ):
        super().__init__(env)
        self.recorder = recorder
        self.env_config = env_config
        self._episode_index = 0
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._episode_start_ms: Optional[float] = None
        self._episode_id: Optional[str] = None

        cfg_payload = (
            env_config.to_dict()
            if hasattr(env_config, "to_dict")
            else dict(env_config)
        )
        meta = RunMeta(
            run_id=recorder.run_id,
            experiment=recorder.experiment,
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            env_config=cfg_payload,
            package_version=_get_package_version(),
        )
        if meta_overrides:
            cfg = meta.to_dict()
            extra: dict[str, Any] = {}
            for key, value in meta_overrides.items():
                if key in cfg:
                    cfg[key] = value
                else:
                    extra[key] = value
            if extra:
                cfg["extra"] = extra
            meta = RunMeta(**cfg)

        sim_meta = _build_simulation_metadata(cfg_payload, meta_overrides)
        if sim_meta is not None:
            cfg2 = meta.to_dict()
            cfg2.setdefault("extra", {})["simulation_metadata"] = sim_meta
            meta = RunMeta(**cfg2)

        self.recorder.write_run_meta(meta)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_index += 1
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._episode_start_ms = time.time() * 1000.0
        self._episode_id = f"ep-{self._episode_index:06d}"
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_steps += 1
        self._episode_reward += float(reward)

        pos = info.get("agent_position") or info.get("agent_xy") or (0, 0)
        dist = float(info.get("distance_to_goal", 0.0))
        seed = info.get("seed")

        obs_summary = None
        if hasattr(obs, "__len__"):
            try:
                if len(obs) > 0:
                    obs_summary = [float(obs[0])]
            except Exception:
                obs_summary = None

        step_rec = StepRecord(
            ts=time.time(),
            run_id=self.recorder.run_id,
            episode_id=self._episode_id or "ep-unknown",
            step=self._episode_steps,
            action=_coerce_int(action),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            agent_position=Position(x=_coerce_int(pos[0]), y=_coerce_int(pos[1])),
            distance_to_goal=dist,
            observation_summary=obs_summary,
            seed=_coerce_int(seed) if seed is not None else None,
        )
        self.recorder.append_step(step_rec)

        if terminated or truncated:
            duration_ms = None
            if self._episode_start_ms is not None:
                duration_ms = (time.time() * 1000.0) - self._episode_start_ms
            ep_rec = EpisodeRecord(
                run_id=self.recorder.run_id,
                episode_id=self._episode_id or "ep-unknown",
                terminated=bool(terminated),
                truncated=bool(truncated),
                total_steps=self._episode_steps,
                total_reward=self._episode_reward,
                final_position=Position(
                    x=_coerce_int(pos[0]), y=_coerce_int(pos[1])
                ),
                final_distance_to_goal=dist,
                duration_ms=duration_ms,
                avg_step_time_ms=(
                    (duration_ms / max(1, self._episode_steps))
                    if duration_ms is not None
                    else None
                ),
            )
            self.recorder.append_episode(ep_rec)

        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
