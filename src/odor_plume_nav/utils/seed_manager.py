"""Utilities for deterministic seeding across the project.

This module provides a very small but fully tested seed management helper.  It
implements a singleton :class:`SeedManager` used throughout the tests.  The
manager exposes classmethod based APIs so it can be conveniently reset between
unit tests while still returning the same instance when instantiated multiple
 times.

The implementation focuses on reproducibility and clarity rather than raw
feature coverage.  Only behaviour exercised in the test-suite is included.
"""
from __future__ import annotations

import contextvars
import os
import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np

try:  # Hydra is optional; tests patch ``GlobalHydra`` when required.
    from hydra.core.global_hydra import GlobalHydra  # type: ignore
except Exception:  # pragma: no cover - hydra is not a runtime dependency
    GlobalHydra = None  # type: ignore

from odor_plume_nav.config.schemas import BaseModel, Field, ConfigDict
from loguru import logger


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class SeedConfig(BaseModel):
    """Pydantic model describing seed initialisation parameters."""

    model_config = ConfigDict(extra="forbid")

    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Global seed applied when seeding random and NumPy.",
    )
    numpy_seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional override for NumPy's legacy and Generator seeds.",
    )
    python_seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional override for the :mod:`random` module seed.",
    )
    auto_seed: bool = Field(
        default=True,
        description="Automatically generate a seed when none is provided.",
    )
    hash_environment: bool = Field(
        default=True,
        description="Include a short hash of the platform in the manager's state.",
    )
    validate_initialization: bool = Field(
        default=True,
        description="Generate a random number after seeding as a smoke test.",
    )
    preserve_state: bool = Field(
        default=False,
        description="Capture RNG state so it can later be restored.",
    )
    log_seed_context: bool = Field(
        default=True,
        description="Bind the current seed to log records via ``logger.configure``.",
    )


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------


@dataclass
class RandomState:
    """Container used when preserving/restoring RNG state."""

    python_state: object
    numpy_state: object
    numpy_generator_state: Optional[dict]
    seed: int


@dataclass
class SeedContext:
    """Light-weight context information used by :func:`get_seed_context`."""

    seed: int


# ---------------------------------------------------------------------------
# Seed manager implementation
# ---------------------------------------------------------------------------


class SeedManager:
    """Singleton responsible for all seeding operations.

    The manager exposes a regular constructor but always returns the same
    instance.  The ``reset`` classmethod drops that instance which allows tests
    to start from a clean slate.
    """

    _instance: Optional["SeedManager"] = None
    _lock = threading.Lock()

    _initialized: bool = False  # toggled when an instance is created
    _current_seed: Optional[int] = None
    _numpy_generator: Optional[np.random.Generator] = None
    _preserve_state: bool = False
    _run_id: Optional[str] = None
    _environment_hash: Optional[str] = None
    _initial_state: Optional[Dict[str, object]] = None
    _log_seed_context: bool = True

    # Thread local storage for seed and numpy generator
    _seed_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
        "seed_manager_seed", default=None
    )
    _numpy_var: contextvars.ContextVar[Optional[np.random.Generator]] = contextvars.ContextVar(
        "seed_manager_numpy", default=None
    )

    def __new__(cls) -> "SeedManager":  # pragma: no cover - tiny wrapper
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._initialized = True
        return cls._instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def current_seed(self) -> Optional[int]:
        return self._seed_var.get()

    @property
    def numpy_generator(self) -> Optional[np.random.Generator]:
        return self._numpy_var.get()

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    @property
    def environment_hash(self) -> Optional[str]:
        return self._environment_hash

    # ------------------------------------------------------------------
    # Core behaviour
    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            cls._current_seed = None
            cls._numpy_generator = None
            cls._seed_var.set(None)
            cls._numpy_var.set(None)
            cls._preserve_state = False
            cls._run_id = None
            cls._environment_hash = None
            cls._initial_state = None
            cls._log_seed_context = True

    # Small helper methods -------------------------------------------------
    def _determine_seed(self, cfg: SeedConfig) -> int:
        """Return the seed to use based on ``cfg``."""
        seed = cfg.seed
        if seed is None:
            if not cfg.auto_seed:
                raise ValueError("Seed value required when auto_seed=False")
            try:
                seed = int.from_bytes(os.urandom(8), "little") & 0xFFFFFFFF
            except Exception as exc:  # pragma: no cover - os.urandom failure
                raise RuntimeError("Auto seed generation failed") from exc
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed out of range")
        return seed

    def _initialize_generators(self, py_seed: int, np_seed: int) -> None:
        random.seed(py_seed)
        np.random.seed(np_seed)
        gen = np.random.default_rng(np_seed)
        type(self)._numpy_generator = gen
        self._numpy_var.set(gen)
        type(self)._current_seed = py_seed
        self._seed_var.set(py_seed)

    def _load_config_from_hydra(self) -> SeedConfig:
        if GlobalHydra is None:  # pragma: no cover - hydra not installed
            return SeedConfig()
        try:
            gh = GlobalHydra.instance()
            if gh and gh.is_initialized():
                cfg = gh.cfg  # type: ignore[attr-defined]
                if "seed_manager" in cfg:
                    data = dict(cfg["seed_manager"])
                elif "seed" in cfg:
                    data = {"seed": cfg["seed"]}
                else:
                    data = {}
                return SeedConfig(**data)
        except Exception:  # pragma: no cover - defensive
            pass
        return SeedConfig()

    def _setup_logging_context(self) -> None:
        if not self._log_seed_context:
            return
        try:
            logger.configure(
                patcher=lambda record: record["extra"].setdefault("seed", self.current_seed)
            )
        except Exception:  # pragma: no cover - logging failures are non fatal
            pass

    # Public API ----------------------------------------------------------
    def initialize(
        self,
        config: Optional[SeedConfig | Dict[str, int]] = None,
        run_id: Optional[str] = None,
    ) -> int:
        """Initialise all random number generators.

        ``config`` may be a :class:`SeedConfig`, a ``dict`` or ``None``.  When
        ``None`` the function will attempt to load configuration from Hydra's
        global configuration if available.
        """

        with self._lock:
            start = time.perf_counter()
            try:
                if config is None:
                    cfg = self._load_config_from_hydra()
                elif isinstance(config, SeedConfig):
                    cfg = config
                else:
                    cfg = SeedConfig(**dict(config))

                seed = self._determine_seed(cfg)
                py_seed = cfg.python_seed if cfg.python_seed is not None else seed
                np_seed = cfg.numpy_seed if cfg.numpy_seed is not None else seed

                self._initialize_generators(py_seed, np_seed)

                self._preserve_state = cfg.preserve_state
                self._run_id = run_id or f"run_{seed:08x}"
                self._log_seed_context = cfg.log_seed_context

                env: Optional[str] = None
                if cfg.hash_environment:
                    import platform, hashlib

                    try:
                        env = platform.platform()
                        self._environment_hash = hashlib.sha256(env.encode()).hexdigest()[:8]
                    except Exception:
                        self._environment_hash = None

                self._setup_logging_context()

                logger.debug("Using configured seed")
                logger.debug("Initialized random generators")

                if cfg.validate_initialization:
                    _ = random.random()
                    _ = np.random.random()
                    logger.debug("Random state validation samples")

                if cfg.log_seed_context:
                    logger.debug("Seed context binding enabled")

                if self._preserve_state:
                    self._initial_state = self.get_state()

                init_time_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    f"Seed manager initialized successfully (seed={seed}, run_id={self._run_id})",
                    extra={
                        "seed": seed,
                        "run_id": self._run_id,
                        "initialization_time_ms": init_time_ms,
                        "environment_hash": self._environment_hash,
                        "numpy_version": np.__version__,
                        "platform": env,
                    },
                )

                return seed
            except Exception as exc:
                # clean state so subsequent calls can retry
                type(self)._current_seed = None
                type(self)._numpy_generator = None
                self._seed_var.set(None)
                self._numpy_var.set(None)
                self._run_id = None
                self._environment_hash = None
                logger.error(
                    "Seed manager initialization failed",
                    extra={"error_type": type(exc).__name__},
                )
                raise RuntimeError("Seed manager initialization failed") from exc
            finally:
                duration = (time.perf_counter() - start) * 1000
                if duration > 100:
                    logger.warning(
                        f"Seed initialization exceeded performance requirement: {duration:.2f}ms"
                    )

    # ------------------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        self._initialize_generators(seed, seed)

    def get_state(self) -> Optional[Dict[str, object]]:
        if not self._preserve_state:
            return None
        state = {
            "python_state": random.getstate(),
            "numpy_legacy_state": np.random.get_state(),
            "seed": self.current_seed,
            "timestamp": time.time(),
        }
        gen = self.numpy_generator
        if gen is not None:
            state["numpy_generator_state"] = gen.bit_generator.state
        return state

    capture_state = get_state  # alias expected by some helpers

    def restore_state(self, state: Dict[str, object]) -> None:
        if not self._preserve_state:
            raise RuntimeError("State preservation not enabled")
        try:
            random.setstate(state["python_state"])  # type: ignore[arg-type]
            np.random.set_state(state["numpy_legacy_state"])  # type: ignore[arg-type]
            if self._numpy_generator is not None and "numpy_generator_state" in state:
                self._numpy_generator.bit_generator.state = state["numpy_generator_state"]  # type: ignore[index]
            seed = state.get("seed")
            if isinstance(seed, int):
                type(self)._current_seed = seed
                self._seed_var.set(seed)
                if self._numpy_generator is not None:
                    self._numpy_var.set(self._numpy_generator)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("State restoration failed") from exc

    @contextmanager
    def temporary_seed(self, seed: int) -> Iterator[None]:
        if not self._preserve_state:
            raise RuntimeError("Temporary seed requires preserve_state=True")
        prev_seed = self.current_seed
        state = self.get_state()
        self.set_seed(seed)
        try:
            yield
        finally:
            if state is not None:
                self.restore_state(state)
                type(self)._current_seed = prev_seed
                self._seed_var.set(prev_seed)
                if self._numpy_generator is not None:
                    self._numpy_var.set(self._numpy_generator)

    def generate_experiment_seeds(
        self, count: int, base_seed: Optional[int] = None
    ) -> List[int]:
        if self.current_seed is None:
            raise RuntimeError("No seed available for experiment seed generation")
        seed = self.current_seed if base_seed is None else base_seed
        rng = np.random.default_rng(seed)
        return [int(x) for x in rng.integers(0, 2**32, size=count, dtype=np.uint32)]

    def validate_reproducibility(
        self, reference: Dict[str, float], *, tolerance: float = 1e-9
    ) -> bool:
        try:
            if self._initial_state is not None:
                self.restore_state(self._initial_state)
            else:
                if self.current_seed is not None:
                    self.set_seed(self.current_seed)
            results: Dict[str, float] = {}
            if "python_random" in reference:
                results["python_random"] = random.random()
            if "numpy_legacy" in reference:
                results["numpy_legacy"] = float(np.random.random())
            if "numpy_generator" in reference and self.numpy_generator is not None:
                results["numpy_generator"] = float(self.numpy_generator.random())
            for key, ref_val in reference.items():
                val = results.get(key)
                if val is None or abs(val - ref_val) > tolerance:
                    logger.error(
                        "Reproducibility check failed", extra={"key": key}
                    )
                    return False
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Reproducibility check failed", extra={"error_type": type(exc).__name__}
            )
            return False


# ---------------------------------------------------------------------------
# Convenience helpers mirroring the original API used in ``utils.__init__``
# ---------------------------------------------------------------------------

_global_manager: Optional[SeedManager] = None


def get_seed_manager() -> SeedManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = SeedManager()
    return _global_manager


def set_global_seed(
    seed: Optional[int] = None, *, config: Optional[SeedConfig | Dict[str, int]] = None
) -> int:
    data = {} if config is None else (
        config.model_dump() if isinstance(config, SeedConfig) else dict(config)
    )
    if seed is not None:
        data["seed"] = seed
    manager = get_seed_manager()
    return manager.initialize(data)


def get_current_seed() -> Optional[int]:
    return SeedManager._seed_var.get()


def get_numpy_generator() -> Optional[np.random.Generator]:
    return SeedManager._numpy_var.get()


def setup_global_seed(config: Optional[SeedConfig | Dict[str, int]] = None) -> int:
    return set_global_seed(config=config)


@contextmanager
def get_seed_context(seed: int) -> Iterator[SeedContext]:
    manager = get_seed_manager()
    with manager.temporary_seed(seed):
        yield SeedContext(seed=seed)


__all__ = [
    "SeedConfig",
    "SeedManager",
    "SeedContext",
    "RandomState",
    "set_global_seed",
    "setup_global_seed",
    "get_seed_manager",
    "get_current_seed",
    "get_numpy_generator",
    "get_seed_context",
]
