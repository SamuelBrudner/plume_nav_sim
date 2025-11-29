"""Analysis-ready data capture utilities.

This package provides:
- Pydantic schemas for run/step/episode records
- A high-performance gzipped JSONL writer with buffered appends
- A run recorder that manages a results directory and record writing
- An optional Gymnasium wrapper to capture steps/episodes non-invasively

Note: Logging (loguru) is intentionally orthogonal and not used for data writing.
"""

from .loader import ReplayArtifacts, ReplayLoadError, load_replay_artifacts
from .recorder import RunRecorder
from .replay import ReplayConsistencyError, ReplayEngine, ReplayEnvFactory
from .schemas import SCHEMA_VERSION, EpisodeRecord, RunMeta, StepRecord
from .writer import JSONLGzWriter

__all__ = [
    "RunMeta",
    "StepRecord",
    "EpisodeRecord",
    "SCHEMA_VERSION",
    "JSONLGzWriter",
    "RunRecorder",
    "ReplayArtifacts",
    "ReplayLoadError",
    "ReplayConsistencyError",
    "ReplayEngine",
    "ReplayEnvFactory",
    "load_replay_artifacts",
]
