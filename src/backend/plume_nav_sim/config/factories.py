from ..actions import DiscreteGridActions, OrientedGridActions, OrientedRunTumbleActions
from ..core.geometry import Coordinates, GridSize
from ..envs import ComponentBasedEnvironment
from ..observations import AntennaeArraySensor, ConcentrationSensor, WindVectorSensor
from ..plume.gaussian import GaussianPlume
from ..rewards import SparseGoalReward, StepPenaltyReward
from ..wind_field import ConstantWindField
from .component_configs import (
    ActionConfig,
    EnvironmentConfig,
    ObservationConfig,
    PlumeConfig,
    RewardConfig,
    WindConfig,
)

__all__ = [
    "create_action_processor",
    "create_observation_model",
    "create_reward_function",
    "create_concentration_field",
    "create_wind_field",
    "create_environment_from_config",
]


def create_action_processor(config: ActionConfig):
    if config.type == "discrete":
        return DiscreteGridActions(step_size=config.step_size)
    elif config.type == "oriented":
        return OrientedGridActions(step_size=config.step_size)
    elif config.type == "run_tumble":
        return OrientedRunTumbleActions(step_size=config.step_size)
    else:
        raise ValueError(
            f"Invalid action type: {config.type}. Must be 'discrete', 'oriented', or 'run_tumble'."
        )


def create_observation_model(config: ObservationConfig):
    if config.type == "concentration":
        return ConcentrationSensor()
    elif config.type == "antennae":
        return AntennaeArraySensor(
            n_sensors=config.n_sensors,
            sensor_angles=config.sensor_angles,
            sensor_distance=config.sensor_distance,
        )
    elif config.type == "wind_vector":
        return WindVectorSensor(noise_std=float(config.noise_std))
    else:
        raise ValueError(
            f"Invalid observation type: {config.type}. "
            f"Must be 'concentration', 'antennae', or 'wind_vector'."
        )


def create_reward_function(config: RewardConfig, goal_location: Coordinates):
    if config.type == "sparse":
        return SparseGoalReward(
            goal_position=goal_location,
            goal_radius=config.goal_radius,
        )
    elif config.type == "step_penalty":
        return StepPenaltyReward(
            goal_position=goal_location,
            goal_radius=config.goal_radius,
            goal_reward=config.goal_reward,
            step_penalty=config.step_penalty,
        )
    else:
        raise ValueError(
            f"Invalid reward type: {config.type}. Must be 'sparse' or 'step_penalty'."
        )


def create_concentration_field(
    config: PlumeConfig, grid_size: GridSize, goal_location: Coordinates
):
    """Create concentration field from configuration."""
    return GaussianPlume(
        grid_size=grid_size,
        source_location=goal_location,
        sigma=config.sigma,
    )


def create_wind_field(config: WindConfig | None):
    if config is None:
        return None

    if config.type != "constant":
        raise ValueError(
            f"Invalid wind type: {config.type}. Only 'constant' is supported."
        )

    return ConstantWindField(
        direction_deg=config.direction_deg,
        speed=config.speed,
        vector=config.vector,
    )


def create_environment_from_config(
    config: EnvironmentConfig,
) -> ComponentBasedEnvironment:
    # Convert tuples to proper types
    grid_size = GridSize(width=config.grid_size[0], height=config.grid_size[1])
    goal_location = Coordinates(x=config.goal_location[0], y=config.goal_location[1])

    start_location = None
    if config.start_location:
        start_location = Coordinates(
            x=config.start_location[0], y=config.start_location[1]
        )

    # Create components from configs
    action_processor = create_action_processor(config.action)
    observation_model = create_observation_model(config.observation)
    reward_function = create_reward_function(config.reward, goal_location)
    concentration_field = create_concentration_field(
        config.plume, grid_size, goal_location
    )
    wind_field = create_wind_field(config.wind)

    # Assemble environment
    return ComponentBasedEnvironment(
        action_processor=action_processor,
        observation_model=observation_model,
        reward_function=reward_function,
        concentration_field=concentration_field,
        wind_field=wind_field,
        grid_size=grid_size,
        max_steps=config.max_steps,
        goal_location=goal_location,
        goal_radius=config.reward.goal_radius,
        start_location=start_location,
        render_mode=config.render_mode,
    )
