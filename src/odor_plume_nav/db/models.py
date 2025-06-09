"""
Database model definitions for the odor plume navigation system.

This module provides SQLAlchemy declarative base classes and table schemas for 
trajectory recording, experiment metadata storage, and collaborative research 
data sharing. The models are designed to integrate seamlessly with optional 
database persistence capabilities while maintaining zero impact on file-based 
operations.

Key Features:
- Modern SQLAlchemy 2.0+ patterns with declarative base
- Comprehensive data models for experiments, simulations, and trajectories
- Proper type annotations and relationship definitions
- Integration with Pydantic for model validation and serialization
- Optimized indexes for efficient querying of simulation data
- Support for multi-backend database systems (SQLite, PostgreSQL, MySQL)

Design Principles:
- Separation of concerns between database schema and session management
- Optional activation without affecting default file-based workflows
- Research-focused schema supporting experiment tracking and metadata
- Performance-optimized for large-scale trajectory datasets
- Type-safe interfaces compatible with scientific Python ecosystem
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import json
import uuid

from sqlalchemy import (
    String, Integer, Float, DateTime, Text, LargeBinary, Boolean,
    JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint,
    event, select
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
    validates, Session
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.dialects.mysql import JSON as MySQLJSON

try:
    from pydantic import BaseModel, Field, ConfigDict, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


# ============================================================================
# DECLARATIVE BASE AND MIXINS
# ============================================================================

class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base for odor plume navigation database models.
    
    Provides common functionality and type mapping for all database tables
    with support for modern SQLAlchemy 2.0+ patterns and cross-database
    compatibility.
    """
    
    # Type annotation mappings for better IDE support and type safety
    type_annotation_map = {
        str: String,
        int: Integer,
        float: Float,
        bool: Boolean,
        datetime: DateTime,
        dict: JSON,
        list: JSON,
    }


class TimestampMixin:
    """
    Mixin providing automatic timestamp management for database records.
    
    Adds created_at and updated_at fields with automatic population
    on insert and update operations.
    """
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when record was created"
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Timestamp when record was last updated"
    )


class MetadataMixin:
    """
    Mixin providing flexible metadata storage for database records.
    
    Adds a JSON field for storing arbitrary key-value metadata
    with proper serialization and type conversion support.
    """
    
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Flexible JSON metadata storage"
    )
    
    @validates('metadata_')
    def validate_metadata(self, key: str, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate and serialize metadata content."""
        if value is None:
            return None
        
        if not isinstance(value, dict):
            raise ValueError("Metadata must be a dictionary")
        
        # Ensure all values are JSON serializable
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Metadata must be JSON serializable: {e}")
        
        return value


# ============================================================================
# CORE DATABASE MODELS
# ============================================================================

class Experiment(Base, TimestampMixin, MetadataMixin):
    """
    Database model for experiment metadata and configuration storage.
    
    Represents a complete experimental setup with associated configuration,
    status tracking, and descriptive metadata for research organization
    and collaboration.
    
    Attributes:
        experiment_id: Unique identifier for the experiment
        name: Human-readable experiment name
        description: Detailed experiment description
        configuration: Full experiment configuration as JSON
        status: Current experiment status (pending, running, completed, failed)
        researcher: Name or identifier of the researcher conducting the experiment
        tags: List of tags for experiment categorization
        notes: Additional research notes and observations
    """
    
    __tablename__ = 'experiments'
    
    # Primary key and identification
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="Unique experiment identifier (UUID)"
    )
    
    # Core experiment metadata
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable experiment name"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Detailed experiment description and objectives"
    )
    
    # Configuration and parameters
    configuration: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        comment="Complete experiment configuration as JSON"
    )
    
    # Status and workflow management
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default='pending',
        comment="Current experiment status"
    )
    
    # Research metadata
    researcher: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Researcher name or identifier"
    )
    
    tags: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Experiment categorization tags"
    )
    
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Additional research notes and observations"
    )
    
    # Timing information
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when experiment execution started"
    )
    
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when experiment execution completed"
    )
    
    # Relationships
    simulations: Mapped[List["Simulation"]] = relationship(
        "Simulation",
        back_populates="experiment",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="valid_experiment_status"
        ),
        Index("idx_experiment_name", "name"),
        Index("idx_experiment_status", "status"),
        Index("idx_experiment_researcher", "researcher"),
        Index("idx_experiment_created", "created_at"),
        Index("idx_experiment_started", "started_at"),
        Index("idx_experiment_completed", "completed_at"),
    )
    
    @validates('status')
    def validate_status(self, key: str, value: str) -> str:
        """Validate experiment status values."""
        valid_statuses = {'pending', 'running', 'completed', 'failed', 'cancelled'}
        if value not in valid_statuses:
            raise ValueError(f"Invalid status '{value}'. Must be one of: {valid_statuses}")
        return value
    
    @validates('tags')
    def validate_tags(self, key: str, value: Optional[List[str]]) -> Optional[List[str]]:
        """Validate experiment tags format."""
        if value is None:
            return None
        
        if not isinstance(value, list):
            raise ValueError("Tags must be a list of strings")
        
        if not all(isinstance(tag, str) for tag in value):
            raise ValueError("All tags must be strings")
        
        return value
    
    @hybrid_property
    def duration(self) -> Optional[float]:
        """Calculate experiment duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        return self.status in {'pending', 'running'}
    
    def mark_started(self) -> None:
        """Mark experiment as started with current timestamp."""
        self.status = 'running'
        self.started_at = datetime.now(timezone.utc)
    
    def mark_completed(self) -> None:
        """Mark experiment as completed with current timestamp."""
        self.status = 'completed'
        self.completed_at = datetime.now(timezone.utc)
    
    def mark_failed(self, error_message: Optional[str] = None) -> None:
        """Mark experiment as failed with optional error information."""
        self.status = 'failed'
        self.completed_at = datetime.now(timezone.utc)
        if error_message and self.metadata_:
            self.metadata_['error_message'] = error_message
        elif error_message:
            self.metadata_ = {'error_message': error_message}
    
    def __repr__(self) -> str:
        return (f"<Experiment(id='{self.experiment_id}', name='{self.name}', "
                f"status='{self.status}')>")


class Simulation(Base, TimestampMixin, MetadataMixin):
    """
    Database model for individual simulation runs within experiments.
    
    Represents a single simulation execution with timing information,
    agent configuration, and performance metrics for detailed analysis
    of navigation behavior and system performance.
    
    Attributes:
        simulation_id: Unique identifier for the simulation run
        experiment_id: Foreign key linking to parent experiment
        run_number: Sequential run number within the experiment
        agent_count: Number of agents in the simulation
        duration_seconds: Simulation duration in seconds
        parameters: Simulation-specific parameters as JSON
        status: Current simulation status
        performance_summary: Summary performance metrics
    """
    
    __tablename__ = 'simulations'
    
    # Primary key and identification
    simulation_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="Unique simulation identifier (UUID)"
    )
    
    # Foreign key relationship to experiment
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey('experiments.experiment_id', ondelete='CASCADE'),
        nullable=False,
        comment="Parent experiment identifier"
    )
    
    # Simulation metadata
    run_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Sequential run number within experiment"
    )
    
    # Simulation configuration
    agent_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of agents in the simulation"
    )
    
    duration_seconds: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Simulation duration in seconds"
    )
    
    parameters: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        comment="Simulation-specific parameters and configuration"
    )
    
    # Status and timing
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default='pending',
        comment="Current simulation status"
    )
    
    start_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Simulation start timestamp"
    )
    
    end_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Simulation end timestamp"
    )
    
    # Performance summary
    performance_summary: Mapped[Optional[Dict[str, float]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Summary performance metrics for quick access"
    )
    
    # Relationships
    experiment: Mapped["Experiment"] = relationship(
        "Experiment",
        back_populates="simulations"
    )
    
    trajectories: Mapped[List["Trajectory"]] = relationship(
        "Trajectory",
        back_populates="simulation",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    performance_metrics: Mapped[List["PerformanceMetric"]] = relationship(
        "PerformanceMetric",
        back_populates="simulation",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="valid_simulation_status"
        ),
        CheckConstraint("agent_count > 0", name="positive_agent_count"),
        CheckConstraint("duration_seconds IS NULL OR duration_seconds >= 0", name="non_negative_duration"),
        UniqueConstraint("experiment_id", "run_number", name="unique_run_number_per_experiment"),
        Index("idx_simulation_experiment", "experiment_id"),
        Index("idx_simulation_status", "status"),
        Index("idx_simulation_agent_count", "agent_count"),
        Index("idx_simulation_start_time", "start_time"),
        Index("idx_simulation_end_time", "end_time"),
        Index("idx_simulation_duration", "duration_seconds"),
    )
    
    @validates('status')
    def validate_status(self, key: str, value: str) -> str:
        """Validate simulation status values."""
        valid_statuses = {'pending', 'running', 'completed', 'failed', 'cancelled'}
        if value not in valid_statuses:
            raise ValueError(f"Invalid status '{value}'. Must be one of: {valid_statuses}")
        return value
    
    @validates('agent_count')
    def validate_agent_count(self, key: str, value: int) -> int:
        """Validate agent count is positive."""
        if value <= 0:
            raise ValueError("Agent count must be positive")
        return value
    
    @validates('duration_seconds')
    def validate_duration(self, key: str, value: Optional[float]) -> Optional[float]:
        """Validate simulation duration is non-negative."""
        if value is not None and value < 0:
            raise ValueError("Duration must be non-negative")
        return value
    
    @hybrid_property
    def actual_duration(self) -> Optional[float]:
        """Calculate actual simulation duration from timestamps."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if simulation is currently active."""
        return self.status in {'pending', 'running'}
    
    def mark_started(self) -> None:
        """Mark simulation as started with current timestamp."""
        self.status = 'running'
        self.start_time = datetime.now(timezone.utc)
    
    def mark_completed(self, duration: Optional[float] = None) -> None:
        """Mark simulation as completed with duration information."""
        self.status = 'completed'
        self.end_time = datetime.now(timezone.utc)
        if duration is not None:
            self.duration_seconds = duration
        elif self.start_time:
            self.duration_seconds = self.actual_duration
    
    def mark_failed(self, error_message: Optional[str] = None) -> None:
        """Mark simulation as failed with optional error information."""
        self.status = 'failed'
        self.end_time = datetime.now(timezone.utc)
        if error_message and self.metadata_:
            self.metadata_['error_message'] = error_message
        elif error_message:
            self.metadata_ = {'error_message': error_message}
    
    def __repr__(self) -> str:
        return (f"<Simulation(id='{self.simulation_id}', experiment='{self.experiment_id}', "
                f"run={self.run_number}, agents={self.agent_count})>")


class Trajectory(Base, TimestampMixin, MetadataMixin):
    """
    Database model for agent trajectory and sensor data storage.
    
    Stores complete trajectory information for individual agents including
    position history, sensor readings, and associated metadata for detailed
    navigation analysis and research documentation.
    
    Attributes:
        trajectory_id: Unique identifier for the trajectory record
        simulation_id: Foreign key linking to parent simulation
        agent_id: Agent identifier within the simulation
        trajectory_data: Binary-encoded trajectory position data
        sensor_data: Binary-encoded sensor reading data
        step_count: Number of simulation steps recorded
        data_format: Format specification for binary data decoding
        compression: Compression algorithm used for data storage
    """
    
    __tablename__ = 'trajectories'
    
    # Primary key and identification
    trajectory_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="Unique trajectory identifier (UUID)"
    )
    
    # Foreign key relationship to simulation
    simulation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey('simulations.simulation_id', ondelete='CASCADE'),
        nullable=False,
        comment="Parent simulation identifier"
    )
    
    # Agent identification
    agent_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Agent identifier within the simulation"
    )
    
    # Data storage fields
    trajectory_data: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary,
        nullable=True,
        comment="Binary-encoded trajectory position data"
    )
    
    sensor_data: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary,
        nullable=True,
        comment="Binary-encoded sensor reading data"
    )
    
    # Data metadata
    step_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of simulation steps recorded"
    )
    
    data_format: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default='numpy',
        comment="Format specification for binary data decoding"
    )
    
    compression: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="Compression algorithm used for data storage"
    )
    
    # Summary statistics for quick access
    start_position: Mapped[Optional[List[float]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Agent starting position [x, y]"
    )
    
    end_position: Mapped[Optional[List[float]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Agent ending position [x, y]"
    )
    
    total_distance: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Total distance traveled by agent"
    )
    
    max_speed: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Maximum speed achieved by agent"
    )
    
    avg_speed: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Average speed of agent throughout trajectory"
    )
    
    # Relationships
    simulation: Mapped["Simulation"] = relationship(
        "Simulation",
        back_populates="trajectories"
    )
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("agent_id >= 0", name="non_negative_agent_id"),
        CheckConstraint("step_count >= 0", name="non_negative_step_count"),
        CheckConstraint(
            "data_format IN ('numpy', 'json', 'pickle', 'hdf5')",
            name="valid_data_format"
        ),
        CheckConstraint(
            "compression IS NULL OR compression IN ('gzip', 'lzf', 'blosc', 'zstd')",
            name="valid_compression"
        ),
        UniqueConstraint("simulation_id", "agent_id", name="unique_agent_per_simulation"),
        Index("idx_trajectory_simulation", "simulation_id"),
        Index("idx_trajectory_agent", "agent_id"),
        Index("idx_trajectory_step_count", "step_count"),
        Index("idx_trajectory_format", "data_format"),
        Index("idx_trajectory_distance", "total_distance"),
        Index("idx_trajectory_speed", "avg_speed"),
    )
    
    @validates('agent_id')
    def validate_agent_id(self, key: str, value: int) -> int:
        """Validate agent ID is non-negative."""
        if value < 0:
            raise ValueError("Agent ID must be non-negative")
        return value
    
    @validates('step_count')
    def validate_step_count(self, key: str, value: int) -> int:
        """Validate step count is non-negative."""
        if value < 0:
            raise ValueError("Step count must be non-negative")
        return value
    
    @validates('data_format')
    def validate_data_format(self, key: str, value: str) -> str:
        """Validate data format specification."""
        valid_formats = {'numpy', 'json', 'pickle', 'hdf5'}
        if value not in valid_formats:
            raise ValueError(f"Invalid data format '{value}'. Must be one of: {valid_formats}")
        return value
    
    @validates('compression')
    def validate_compression(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate compression algorithm specification."""
        if value is None:
            return None
        
        valid_compression = {'gzip', 'lzf', 'blosc', 'zstd'}
        if value not in valid_compression:
            raise ValueError(f"Invalid compression '{value}'. Must be one of: {valid_compression}")
        return value
    
    @validates('start_position', 'end_position')
    def validate_position(self, key: str, value: Optional[List[float]]) -> Optional[List[float]]:
        """Validate position coordinates format."""
        if value is None:
            return None
        
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("Position must be a list of [x, y] coordinates")
        
        if not all(isinstance(coord, (int, float)) for coord in value):
            raise ValueError("Position coordinates must be numeric")
        
        return value
    
    @validates('total_distance', 'max_speed', 'avg_speed')
    def validate_positive_metrics(self, key: str, value: Optional[float]) -> Optional[float]:
        """Validate that distance and speed metrics are non-negative."""
        if value is not None and value < 0:
            raise ValueError(f"{key} must be non-negative")
        return value
    
    def calculate_summary_statistics(self, trajectory_array: Optional[Any] = None) -> None:
        """
        Calculate and update summary statistics from trajectory data.
        
        Args:
            trajectory_array: Optional NumPy array of trajectory positions.
                             If not provided, attempts to decode trajectory_data.
        """
        if trajectory_array is None and self.trajectory_data is None:
            return
        
        try:
            if trajectory_array is None:
                # Attempt to decode trajectory data (requires NumPy)
                import numpy as np
                import io
                
                data_stream = io.BytesIO(self.trajectory_data)
                if self.compression == 'gzip':
                    import gzip
                    data_stream = gzip.GzipFile(fileobj=data_stream)
                
                trajectory_array = np.load(data_stream)
            
            if len(trajectory_array) > 0:
                self.start_position = trajectory_array[0].tolist()
                self.end_position = trajectory_array[-1].tolist()
                
                # Calculate distances between consecutive points
                if len(trajectory_array) > 1:
                    diffs = np.diff(trajectory_array, axis=0)
                    distances = np.sqrt(np.sum(diffs**2, axis=1))
                    self.total_distance = float(np.sum(distances))
                    self.max_speed = float(np.max(distances))
                    self.avg_speed = float(np.mean(distances))
                else:
                    self.total_distance = 0.0
                    self.max_speed = 0.0
                    self.avg_speed = 0.0
                    
        except Exception:
            # If calculation fails, leave summary statistics as None
            pass
    
    def __repr__(self) -> str:
        return (f"<Trajectory(id='{self.trajectory_id}', simulation='{self.simulation_id}', "
                f"agent={self.agent_id}, steps={self.step_count})>")


class PerformanceMetric(Base, TimestampMixin):
    """
    Database model for simulation performance metrics and analysis results.
    
    Stores quantitative performance measurements and analysis results for
    detailed system performance tracking, optimization, and research insights
    into navigation algorithm effectiveness.
    
    Attributes:
        metric_id: Unique identifier for the performance metric
        simulation_id: Foreign key linking to parent simulation
        metric_name: Name/type of the performance metric
        metric_value: Numerical value of the metric
        metric_unit: Unit of measurement for the metric
        metric_category: Category for metric organization
        context: Additional context information for the metric
        recorded_at: Timestamp when metric was recorded
    """
    
    __tablename__ = 'performance_metrics'
    
    # Primary key and identification
    metric_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="Unique performance metric identifier (UUID)"
    )
    
    # Foreign key relationship to simulation
    simulation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey('simulations.simulation_id', ondelete='CASCADE'),
        nullable=False,
        comment="Parent simulation identifier"
    )
    
    # Metric identification and categorization
    metric_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Name/type of the performance metric"
    )
    
    metric_category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default='general',
        comment="Category for metric organization"
    )
    
    # Metric value and metadata
    metric_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Numerical value of the metric"
    )
    
    metric_unit: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Unit of measurement for the metric"
    )
    
    # Additional context and timing
    context: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional context information for the metric"
    )
    
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Timestamp when metric was recorded"
    )
    
    # Relationships
    simulation: Mapped["Simulation"] = relationship(
        "Simulation",
        back_populates="performance_metrics"
    )
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_metric_simulation", "simulation_id"),
        Index("idx_metric_name", "metric_name"),
        Index("idx_metric_category", "metric_category"),
        Index("idx_metric_value", "metric_value"),
        Index("idx_metric_recorded", "recorded_at"),
        Index("idx_metric_name_category", "metric_name", "metric_category"),
        Index("idx_metric_simulation_name", "simulation_id", "metric_name"),
    )
    
    @validates('context')
    def validate_context(self, key: str, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate context metadata content."""
        if value is None:
            return None
        
        if not isinstance(value, dict):
            raise ValueError("Context must be a dictionary")
        
        # Ensure all values are JSON serializable
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Context must be JSON serializable: {e}")
        
        return value
    
    def __repr__(self) -> str:
        return (f"<PerformanceMetric(id='{self.metric_id}', simulation='{self.simulation_id}', "
                f"name='{self.metric_name}', value={self.metric_value})>")


# ============================================================================
# PYDANTIC INTEGRATION MODELS (OPTIONAL)
# ============================================================================

if PYDANTIC_AVAILABLE:
    
    class ExperimentSchema(BaseModel):
        """Pydantic schema for Experiment model serialization and validation."""
        
        model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
        
        experiment_id: str = Field(..., description="Unique experiment identifier")
        name: str = Field(..., min_length=1, max_length=255, description="Experiment name")
        description: Optional[str] = Field(None, description="Experiment description")
        configuration: Dict[str, Any] = Field(..., description="Experiment configuration")
        status: str = Field(..., description="Experiment status")
        researcher: Optional[str] = Field(None, max_length=255, description="Researcher identifier")
        tags: Optional[List[str]] = Field(None, description="Experiment tags")
        notes: Optional[str] = Field(None, description="Research notes")
        metadata_: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
        created_at: datetime = Field(..., description="Creation timestamp")
        updated_at: datetime = Field(..., description="Last update timestamp")
        started_at: Optional[datetime] = Field(None, description="Start timestamp")
        completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
        
        @model_validator(mode='after')
        def validate_status_consistency(self):
            """Validate status and timestamp consistency."""
            if self.status == 'running' and self.started_at is None:
                raise ValueError("Running experiments must have a start timestamp")
            if self.status in {'completed', 'failed'} and self.completed_at is None:
                raise ValueError("Completed/failed experiments must have a completion timestamp")
            return self
    
    
    class SimulationSchema(BaseModel):
        """Pydantic schema for Simulation model serialization and validation."""
        
        model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
        
        simulation_id: str = Field(..., description="Unique simulation identifier")
        experiment_id: str = Field(..., description="Parent experiment identifier")
        run_number: int = Field(..., ge=1, description="Run number within experiment")
        agent_count: int = Field(..., gt=0, description="Number of agents")
        duration_seconds: Optional[float] = Field(None, ge=0, description="Simulation duration")
        parameters: Dict[str, Any] = Field(..., description="Simulation parameters")
        status: str = Field(..., description="Simulation status")
        performance_summary: Optional[Dict[str, float]] = Field(None, description="Performance summary")
        metadata_: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
        created_at: datetime = Field(..., description="Creation timestamp")
        updated_at: datetime = Field(..., description="Last update timestamp")
        start_time: Optional[datetime] = Field(None, description="Start timestamp")
        end_time: Optional[datetime] = Field(None, description="End timestamp")
    
    
    class TrajectorySchema(BaseModel):
        """Pydantic schema for Trajectory model serialization and validation."""
        
        model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
        
        trajectory_id: str = Field(..., description="Unique trajectory identifier")
        simulation_id: str = Field(..., description="Parent simulation identifier")
        agent_id: int = Field(..., ge=0, description="Agent identifier")
        step_count: int = Field(..., ge=0, description="Number of recorded steps")
        data_format: str = Field(..., description="Data format specification")
        compression: Optional[str] = Field(None, description="Compression algorithm")
        start_position: Optional[List[float]] = Field(None, description="Starting position")
        end_position: Optional[List[float]] = Field(None, description="Ending position")
        total_distance: Optional[float] = Field(None, ge=0, description="Total distance traveled")
        max_speed: Optional[float] = Field(None, ge=0, description="Maximum speed")
        avg_speed: Optional[float] = Field(None, ge=0, description="Average speed")
        metadata_: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
        created_at: datetime = Field(..., description="Creation timestamp")
        updated_at: datetime = Field(..., description="Last update timestamp")
    
    
    class PerformanceMetricSchema(BaseModel):
        """Pydantic schema for PerformanceMetric model serialization and validation."""
        
        model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
        
        metric_id: str = Field(..., description="Unique metric identifier")
        simulation_id: str = Field(..., description="Parent simulation identifier")
        metric_name: str = Field(..., min_length=1, max_length=255, description="Metric name")
        metric_category: str = Field(..., max_length=100, description="Metric category")
        metric_value: float = Field(..., description="Metric value")
        metric_unit: Optional[str] = Field(None, max_length=50, description="Metric unit")
        context: Optional[Dict[str, Any]] = Field(None, description="Metric context")
        recorded_at: datetime = Field(..., description="Recording timestamp")
        created_at: datetime = Field(..., description="Creation timestamp")
        updated_at: datetime = Field(..., description="Last update timestamp")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_all_tables(engine) -> None:
    """
    Create all database tables using the provided SQLAlchemy engine.
    
    Args:
        engine: SQLAlchemy engine instance connected to the target database
    """
    Base.metadata.create_all(bind=engine)


def drop_all_tables(engine) -> None:
    """
    Drop all database tables using the provided SQLAlchemy engine.
    
    Warning: This will permanently delete all data in the database.
    
    Args:
        engine: SQLAlchemy engine instance connected to the target database
    """
    Base.metadata.drop_all(bind=engine)


def get_table_names() -> List[str]:
    """
    Get a list of all table names defined in the database schema.
    
    Returns:
        List of table names as strings
    """
    return list(Base.metadata.tables.keys())


def validate_database_schema(engine) -> Dict[str, bool]:
    """
    Validate that all expected tables exist in the database.
    
    Args:
        engine: SQLAlchemy engine instance connected to the target database
        
    Returns:
        Dictionary mapping table names to existence status
    """
    from sqlalchemy import inspect
    
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    expected_tables = set(Base.metadata.tables.keys())
    
    return {table: table in existing_tables for table in expected_tables}


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Base classes and mixins
    'Base',
    'TimestampMixin',
    'MetadataMixin',
    
    # Core database models
    'Experiment',
    'Simulation',
    'Trajectory',
    'PerformanceMetric',
    
    # Pydantic schemas (if available)
    'ExperimentSchema',
    'SimulationSchema',
    'TrajectorySchema',
    'PerformanceMetricSchema',
    
    # Utility functions
    'create_all_tables',
    'drop_all_tables',
    'get_table_names',
    'validate_database_schema',
    
    # Constants
    'PYDANTIC_AVAILABLE',
]