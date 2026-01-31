from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ActionConfig",
    "ObservationConfig",
    "RewardConfig",
    "PlumeConfig",
    "WindConfig",
    "EnvironmentConfig",
]


class ActionConfig(BaseModel):
    type: Literal["discrete", "oriented", "run_tumble"] = Field(
        default="discrete",
        description=(
            "Action processor type: 'discrete' (4-dir), 'oriented' (3-action), "
            "or 'run_tumble' (2-action)."
        ),
    )
    step_size: int = Field(
        default=1, ge=1, description="Movement step size in grid cells"
    )
    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class ObservationConfig(BaseModel):
    type: Literal["concentration", "antennae", "wind_vector"] = Field(
        default="concentration",
        description=(
            "Observation model type: 'concentration' (single), 'antennae' (array), "
            "or 'wind_vector' (mechanosensory wind)"
        ),
    )
    n_sensors: int = Field(
        default=2, ge=1, description="Number of sensors (for antennae array)"
    )
    sensor_distance: float = Field(
        default=1.0, gt=0.0, description="Distance from agent to sensors (grid cells)"
    )
    sensor_angles: Optional[list[float]] = Field(
        default=None, description="Custom sensor angles in degrees (optional)"
    )
    noise_std: float = Field(
        default=0.0,
        ge=0.0,
        description="Gaussian noise stddev for wind_vector observations",
    )

    @field_validator("sensor_angles")
    @classmethod
    def validate_sensor_angles(cls, v, info):
        """Validate sensor angles match n_sensors if provided."""
        if v is not None:
            n_sensors = info.data.get("n_sensors", 2)
            if len(v) != n_sensors:
                raise ValueError(
                    f"sensor_angles length ({len(v)}) must match n_sensors ({n_sensors})"
                )
        return v

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class WindConfig(BaseModel):
    type: Literal["constant"] = Field(
        default="constant", description="Wind model type (constant vector)"
    )
    direction_deg: float = Field(
        default=0.0, description="Wind direction in degrees (0°=East, 90°=North)"
    )
    speed: float = Field(
        default=1.0,
        ge=0.0,
        description="Wind speed magnitude; ignored if vector provided",
    )
    vector: Optional[tuple[float, float]] = Field(
        default=None, description="Optional explicit wind vector (vx, vy)"
    )

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v):
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("vector must have length 2 (vx, vy)")
        return v

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class RewardConfig(BaseModel):
    type: Literal["sparse", "step_penalty"] = Field(
        default="sparse",
        description="Reward function type: 'sparse' (binary) or 'step_penalty' (time-aware)",
    )
    goal_radius: float = Field(
        default=5.0, gt=0.0, description="Success threshold distance from goal"
    )
    goal_reward: float = Field(
        default=1.0,
        description="Reward granted upon reaching the goal (step_penalty reward)",
    )
    step_penalty: float = Field(
        default=0.01,
        ge=0.0,
        description="Penalty applied each step until goal (step_penalty reward)",
    )

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class PlumeConfig(BaseModel):
    sigma: float = Field(
        default=20.0, gt=0.0, description="Gaussian dispersion parameter (std dev)"
    )
    normalize: bool = Field(
        default=True, description="Normalize field values to [0, 1]"
    )
    enable_caching: bool = Field(
        default=True, description="Enable concentration field caching"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Deprecated; no custom plume parameters are currently supported",
    )

    @field_validator("parameters")
    @classmethod
    def _reject_plume_parameters(cls, v):
        if v:
            raise ValueError(
                "PlumeConfig.parameters is deprecated; no parameters are used"
            )
        return v

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class EnvironmentConfig(BaseModel):
    # Environment settings
    grid_size: tuple[int, int] = Field(
        default=(128, 128), description="Environment dimensions (width, height)"
    )
    goal_location: tuple[int, int] = Field(
        default=(64, 64), description="Target position (x, y)"
    )
    start_location: Optional[tuple[int, int]] = Field(
        default=None, description="Initial agent position (x, y), None = grid center"
    )
    max_steps: int = Field(default=1000, ge=1, description="Episode step limit")
    render_mode: Optional[Literal["rgb_array", "human"]] = Field(
        default=None, description="Rendering mode"
    )

    # Component configs
    action: ActionConfig = Field(
        default_factory=ActionConfig, description="Action processor configuration"
    )
    observation: ObservationConfig = Field(
        default_factory=ObservationConfig, description="Observation model configuration"
    )
    reward: RewardConfig = Field(
        default_factory=RewardConfig, description="Reward function configuration"
    )
    plume: PlumeConfig = Field(
        default_factory=PlumeConfig, description="Plume field configuration"
    )
    wind: Optional[WindConfig] = Field(
        default=None, description="Wind model configuration (None disables wind)"
    )

    @field_validator("grid_size")
    @classmethod
    def validate_grid_size(cls, v):
        """Validate grid size is positive."""
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"grid_size must be positive, got {v}")
        return v

    @field_validator("goal_location", "start_location")
    @classmethod
    def validate_coordinates(cls, v, info):
        """Validate coordinates are within grid bounds."""
        if v is None:
            return v  # start_location can be None
        grid_size = info.data.get("grid_size", (128, 128))
        if v[0] < 0 or v[0] >= grid_size[0] or v[1] < 0 or v[1] >= grid_size[1]:
            raise ValueError(
                f"Coordinates {v} must be within grid bounds (0, 0) to {grid_size}"
            )
        return v

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "grid_size": [128, 128],
                "goal_location": [64, 64],
                "max_steps": 1000,
                "action": {"type": "discrete", "step_size": 1},
                "observation": {"type": "concentration"},
                "reward": {"type": "sparse", "goal_radius": 5.0},
                "plume": {"sigma": 20.0, "normalize": True},
                "wind": {"type": "constant", "direction_deg": 0.0, "speed": 1.0},
            }
        },
    )
