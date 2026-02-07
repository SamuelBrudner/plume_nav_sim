"""Legacy environment compatibility helpers.

This module preserves deprecated wrapper APIs that previously lived in
``plume_nav_sim.envs``.
"""

import time
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium

from .._compat import (
    ComponentError,
    ValidationError,
    validate_coordinates,
    validate_grid_size,
)
from ..constants import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_SOURCE_LOCATION,
)
from ..logging import get_component_logger
from ..registration.register import ENV_ID, is_registered, register_env, unregister_env
from .component_env import ComponentBasedEnvironment
from .factory import create_component_environment
from .plume_env import PlumeEnv, create_plume_env
from .plume_search_env import (
    PlumeSearchEnv,
    create_plume_search_env,
    validate_plume_search_config,
)
from .state import EnvironmentState

ENVIRONMENT_VERSION = "1.0.0"

SUPPORTED_ENVIRONMENTS = [
    "plume_env",
    "plume",
    "PlumeEnv",
    "plume_search",
    "component",
    "PlumeSearchEnv",
]

_module_logger = get_component_logger("envs")

__all__ = [
    "PlumeEnv",
    "create_plume_env",
    "EnvironmentState",
    "ComponentBasedEnvironment",
    "create_component_environment",
    "PlumeSearchEnv",
    "create_plume_search_env",
    "validate_plume_search_config",
    "register_env",
    "unregister_env",
    "is_registered",
    "ENV_ID",
    "DEFAULT_GRID_SIZE",
    "DEFAULT_SOURCE_LOCATION",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_GOAL_RADIUS",
    "create_environment",
    "make_environment",
    "get_environment_info",
]


def create_environment(  # noqa: C901
    env_type: Optional[str] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    source_location: Optional[Tuple[int, int]] = None,
    max_steps: Optional[int] = None,
    goal_radius: Optional[float] = None,
    render_mode: Optional[str] = None,
    env_options: Optional[Dict[str, Any]] = None,
) -> gymnasium.Env:
    creation_start_time = time.perf_counter()

    try:
        effective_env_type = env_type or "plume_env"

        _module_logger.debug(
            f"Creating environment: type={effective_env_type}, grid_size={grid_size}"
        )

        if effective_env_type not in SUPPORTED_ENVIRONMENTS:
            raise ValidationError(
                f"Unsupported environment type: {effective_env_type}",
                parameter_name="env_type",
                parameter_value=effective_env_type,
                expected_format=f"One of: {SUPPORTED_ENVIRONMENTS}",
            )

        effective_grid_size = grid_size if grid_size is not None else DEFAULT_GRID_SIZE
        effective_source_location = (
            source_location if source_location is not None else DEFAULT_SOURCE_LOCATION
        )
        effective_max_steps = max_steps if max_steps is not None else DEFAULT_MAX_STEPS
        effective_goal_radius = (
            goal_radius if goal_radius is not None else DEFAULT_GOAL_RADIUS
        )
        effective_render_mode = render_mode if render_mode is not None else "rgb_array"

        try:
            validate_grid_size(effective_grid_size)
        except ValidationError as exc:
            raise ValidationError(
                f"Invalid grid_size: {effective_grid_size}",
                parameter_name="grid_size",
                expected_format="Tuple of positive integers (width, height)",
            ) from exc

        try:
            validate_coordinates(effective_source_location)
        except ValidationError as exc:
            raise ValidationError(
                f"Invalid source_location: {effective_source_location}",
                parameter_name="source_location",
                expected_format="Tuple of non-negative coordinates (x, y)",
            ) from exc

        if (
            effective_source_location[0] >= effective_grid_size[0]
            or effective_source_location[1] >= effective_grid_size[1]
        ):
            raise ValidationError(
                f"Source location {effective_source_location} outside grid bounds {effective_grid_size}",
                parameter_name="source_location",
            )

        if not isinstance(effective_max_steps, int) or effective_max_steps <= 0:
            raise ValidationError(
                f"Invalid max_steps: {effective_max_steps}",
                parameter_name="max_steps",
                expected_format="Positive integer",
            )

        if (
            not isinstance(effective_goal_radius, (int, float))
            or effective_goal_radius < 0
        ):
            raise ValidationError(
                f"Invalid goal_radius: {effective_goal_radius}",
                parameter_name="goal_radius",
                expected_format="Non-negative number",
            )

        if effective_env_type in ("plume_env", "plume", "PlumeEnv"):
            plume_env_kwargs = dict(env_options or {})
            plume_type = plume_env_kwargs.pop("plume_type", "gaussian")
            environment = create_plume_env(
                plume_type=plume_type,
                grid_size=effective_grid_size,
                source_location=effective_source_location,
                max_steps=effective_max_steps,
                goal_radius=effective_goal_radius,
                render_mode=effective_render_mode,
                **plume_env_kwargs,
            )
        elif effective_env_type in ("plume_search", "PlumeSearchEnv"):
            warnings.warn(
                "env_type 'plume_search' is deprecated; use 'plume_env' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            environment = create_plume_search_env(
                grid_size=effective_grid_size,
                source_location=effective_source_location,
                max_steps=effective_max_steps,
                goal_radius=effective_goal_radius,
                render_mode=effective_render_mode,
                env_options=env_options,
            )
        elif effective_env_type == "component":
            warnings.warn(
                "env_type 'component' is deprecated; use 'plume_env' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            environment = create_component_environment(
                grid_size=effective_grid_size,
                goal_location=effective_source_location,
                max_steps=effective_max_steps,
                goal_radius=effective_goal_radius,
                render_mode=effective_render_mode,
            )
        else:
            raise ComponentError(
                f"Environment type {effective_env_type} not implemented",
                component_name="create_environment",
                operation_name="environment_creation",
            )

        creation_time_ms = (time.perf_counter() - creation_start_time) * 1000

        _module_logger.info(
            f"Environment created successfully: type={effective_env_type}, "
            f"grid={effective_grid_size[0]}x{effective_grid_size[1]}, "
            f"source=({effective_source_location[0]},{effective_source_location[1]}), "
            f"max_steps={effective_max_steps}, creation_time={creation_time_ms:.2f}ms"
        )

        if creation_time_ms > 100:  # 100ms threshold
            _module_logger.warning(
                f"Environment creation took {creation_time_ms:.2f}ms - consider optimization"
            )

        return environment

    except (ValidationError, ComponentError):
        raise
    except Exception as e:
        creation_time_ms = (time.perf_counter() - creation_start_time) * 1000

        _module_logger.error(
            f"Environment creation failed: {e}, creation_time={creation_time_ms:.2f}ms"
        )

        raise ComponentError(
            f"Environment creation failed: {e}",
            component_name="create_environment",
            operation_name="environment_creation",
        ) from e


def make_environment(  # noqa: C901
    env_type: Optional[str] = None,
    env_config: Optional[Dict[str, Any]] = None,
    auto_register: bool = True,
    force_reregister: bool = False,
) -> gymnasium.Env:
    make_start_time = time.perf_counter()

    try:
        effective_env_type = env_type or "plume_search"
        effective_env_config = env_config or {}

        _module_logger.debug(
            f"Making environment: type={effective_env_type}, auto_register={auto_register}"
        )

        if effective_env_type not in SUPPORTED_ENVIRONMENTS:
            raise ValidationError(
                f"Unsupported environment type: {effective_env_type}",
                parameter_name="env_type",
                parameter_value=effective_env_type,
            )

        currently_registered = is_registered(ENV_ID)

        if auto_register and not currently_registered:
            _module_logger.info(f"Auto-registering environment: {ENV_ID}")

            registration_params = {
                "grid_size": effective_env_config.get("grid_size"),
                "source_location": effective_env_config.get("source_location"),
                "max_steps": effective_env_config.get("max_steps"),
                "goal_radius": effective_env_config.get("goal_radius"),
            }

            registered_env_id = register_env(
                env_id=ENV_ID,
                kwargs=registration_params,
                force_reregister=force_reregister,
            )

            if registered_env_id != ENV_ID:
                _module_logger.warning(
                    f"Registration returned different ID: {registered_env_id} != {ENV_ID}"
                )

        elif force_reregister and currently_registered:
            _module_logger.info(f"Force re-registering environment: {ENV_ID}")

            unregister_env(ENV_ID)

            registration_params = {
                "grid_size": effective_env_config.get("grid_size"),
                "source_location": effective_env_config.get("source_location"),
                "max_steps": effective_env_config.get("max_steps"),
                "goal_radius": effective_env_config.get("goal_radius"),
            }

            register_env(
                env_id=ENV_ID, kwargs=registration_params, force_reregister=True
            )

        try:
            environment = gymnasium.make(ENV_ID)
            _module_logger.debug(f"Environment created via gym.make(): {ENV_ID}")
        except Exception as make_error:
            _module_logger.error(f"gym.make() failed for {ENV_ID}: {make_error}")

            _module_logger.warning("Falling back to direct environment creation")
            environment = create_environment(
                env_type=effective_env_type, **effective_env_config
            )

        if not isinstance(environment, gymnasium.Env):
            raise ComponentError(
                f"Created environment is not gymnasium.Env instance: {type(environment)}",
                component_name="make_environment",
                operation_name="environment_validation",
            )

        try:
            test_obs, test_info = environment.reset()
            environment.close()
            _module_logger.debug("Environment functionality validated successfully")
        except Exception as test_error:
            _module_logger.warning(
                f"Environment functionality test failed: {test_error}"
            )

        make_time_ms = (time.perf_counter() - make_start_time) * 1000

        _module_logger.info(
            f"Environment ready: type={effective_env_type}, registered={currently_registered}, "
            f"make_time={make_time_ms:.2f}ms"
        )

        if make_time_ms > 200:  # 200ms threshold for make_environment
            _module_logger.warning(
                f"make_environment took {make_time_ms:.2f}ms - consider optimization"
            )

        return environment

    except (ValidationError, ComponentError):
        raise
    except Exception as e:
        make_time_ms = (time.perf_counter() - make_start_time) * 1000

        _module_logger.error(
            f"make_environment failed: {e}, make_time={make_time_ms:.2f}ms"
        )

        raise ComponentError(
            f"Environment creation and registration failed: {e}",
            component_name="make_environment",
            operation_name="environment_creation",
        ) from e


def get_environment_info(
    env_type: Optional[str] = None,
    include_examples: bool = False,
    include_performance_info: bool = False,
) -> Dict[str, Any]:
    info_start_time = time.perf_counter()

    try:
        effective_env_type = env_type

        _module_logger.debug(
            f"Gathering environment info: type={effective_env_type}, examples={include_examples}"
        )

        environment_info = {
            "query_timestamp": time.time(),
            "module_version": ENVIRONMENT_VERSION,
            "supported_environments": SUPPORTED_ENVIRONMENTS.copy(),
            "default_environment_type": "plume_env",
            "total_environment_types": len(SUPPORTED_ENVIRONMENTS),
        }

        environment_info["registration_status"] = {
            "registered": is_registered(),
            "environment_available": is_registered(),
        }

        environment_info["default_configuration"] = {
            "grid_size": DEFAULT_GRID_SIZE,
            "source_location": DEFAULT_SOURCE_LOCATION,
            "max_steps": DEFAULT_MAX_STEPS,
            "goal_radius": DEFAULT_GOAL_RADIUS,
            "supported_render_modes": ["rgb_array", "human"],
            "action_space_type": "Discrete(4)",
            "observation_space_type": "Box(shape=(1,), dtype=float32)",
        }

        environment_info["parameter_constraints"] = {
            "grid_size_range": "Tuple of positive integers, recommended (32,32) to (512,512)",
            "source_location_constraint": "Must be within grid bounds (0,0) to (width-1,height-1)",
            "max_steps_range": "Positive integer, typically 100 to 10000",
            "goal_radius_range": "Non-negative float, 0 means exact location required",
            "memory_estimate": "Approximately (width*height*4) bytes for concentration field",
        }

        if effective_env_type:
            if effective_env_type in ("plume_env", "plume", "PlumeEnv"):
                environment_info["plume_env_details"] = {
                    "description": "Flattened plume environment with injectable components",
                    "observation_type": "Sensor model output (default: concentration)",
                    "action_type": "Action model output (default: discrete grid actions)",
                    "reward_structure": "Reward function output (default: sparse goal)",
                    "termination_condition": "Agent reaches source location within goal_radius",
                    "truncation_condition": "Episode exceeds max_steps limit",
                    "plume_model": "Gaussian or video plume backends",
                }
            elif effective_env_type in ("plume_search", "PlumeSearchEnv"):
                environment_info["plume_search_details"] = {
                    "description": "Legacy plume search wrapper environment (deprecated)",
                    "observation_type": "Concentration value at agent position",
                    "action_type": "Discrete cardinal directions (Up, Right, Down, Left)",
                    "reward_structure": "Sparse reward (1.0 for goal, 0.0 otherwise)",
                    "termination_condition": "Agent reaches source location within goal_radius",
                    "truncation_condition": "Episode exceeds max_steps limit",
                    "plume_model": "Static Gaussian distribution with configurable dispersion",
                    "deprecated": True,
                }
            elif effective_env_type == "component":
                environment_info["component_details"] = {
                    "description": "Component-based environment (deprecated)",
                    "deprecated": True,
                }
            else:
                environment_info["environment_type_error"] = (
                    f"Unknown environment type: {effective_env_type}"
                )

        if include_examples:
            environment_info["usage_examples"] = {
                "basic_creation": {
                    "description": "Create environment with default parameters",
                    "code": "env = create_environment()\nobs, info = env.reset()\naction = env.action_space.sample()\nobs, reward, terminated, truncated, info = env.step(action)",
                },
                "custom_configuration": {
                    "description": "Create environment with custom parameters",
                    "code": "env = create_environment(\n    grid_size=(256, 256),\n    source_location=(128, 128),\n    max_steps=2000,\n    goal_radius=5.0\n)",
                },
                "gym_make_usage": {
                    "description": "Use with gym.make() after registration",
                    "code": 'import gymnasium as gym\nregister_env()  # Ensure registration\nenv = gym.make("PlumeNav-v0")',
                },
                "research_workflow": {
                    "description": "Typical research workflow with validation",
                    "code": "env = make_environment(auto_register=True)\nfor episode in range(100):\n    obs, info = env.reset(seed=episode)\n    done = False\n    while not done:\n        action = policy.predict(obs)\n        obs, reward, terminated, truncated, info = env.step(action)\n        done = terminated or truncated",
                },
            }

        if include_performance_info:
            environment_info["performance_information"] = {
                "target_step_latency_ms": 1.0,
                "target_reset_latency_ms": 10.0,
                "target_render_latency_ms": 5.0,
                "memory_usage_estimates": {
                    "default_grid_128x128": "65MB (estimated)",
                    "large_grid_512x512": "1GB (estimated)",
                    "small_grid_64x64": "16MB (estimated)",
                },
                "optimization_recommendations": [
                    "Use default grid size (128,128) for balanced performance",
                    "Consider smaller grids for faster training iterations",
                    "Enable performance monitoring for production use",
                    "Use rgb_array mode for programmatic visualization",
                ],
                "benchmark_expectations": {
                    "steps_per_second_target": ">1000 with default configuration",
                    "episode_creation_time": "<10ms for reset operations",
                    "memory_footprint": "<100MB for production environments",
                },
            }

        environment_info["visualization_capabilities"] = {
            "rgb_array_mode": {
                "description": "Returns RGB array for programmatic use",
                "output_type": "numpy.ndarray with shape (height, width, 3)",
                "use_cases": [
                    "Algorithm visualization",
                    "Automated analysis",
                    "Video generation",
                ],
            },
            "human_mode": {
                "description": "Interactive matplotlib visualization",
                "output_type": "None (displays window)",
                "use_cases": [
                    "Interactive debugging",
                    "Manual inspection",
                    "Research presentation",
                ],
            },
        }

        environment_info["troubleshooting"] = {
            "common_issues": [
                "Registration not found: Call register_env() before gym.make()",
                "Memory issues with large grids: Reduce grid_size parameter",
                "Slow performance: Check grid_size and enable performance monitoring",
                "Import errors: Ensure all dependencies are installed with correct versions",
            ],
            "validation_checklist": [
                "Verify grid_size dimensions are positive integers",
                "Ensure source_location is within grid bounds",
                "Check max_steps is reasonable for training time",
                "Confirm goal_radius is non-negative",
                "Validate render_mode is supported",
            ],
            "debugging_tools": [
                "Use get_environment_info() for full status",
                "Check is_registered() for registration status",
                "Enable debug logging with logging.basicConfig(level=logging.DEBUG)",
                "Use environment.validate_environment_integrity() for health checks",
            ],
        }

        info_time_ms = (time.perf_counter() - info_start_time) * 1000
        environment_info["compilation_time_ms"] = info_time_ms

        _module_logger.debug(
            f"Environment info compiled: info_time={info_time_ms:.2f}ms"
        )

        return environment_info

    except Exception as e:
        info_time_ms = (time.perf_counter() - info_start_time) * 1000

        _module_logger.error(
            f"get_environment_info failed: {e}, info_time={info_time_ms:.2f}ms"
        )

        return {
            "query_timestamp": time.time(),
            "error": str(e),
            "compilation_time_ms": info_time_ms,
            "supported_environments": SUPPORTED_ENVIRONMENTS.copy(),
            "module_version": ENVIRONMENT_VERSION,
            "fallback_info": "Basic information only due to error",
        }


def _validate_environment_parameters(  # noqa: C901
    grid_size: Optional[Tuple[int, int]] = None,
    source_location: Optional[Tuple[int, int]] = None,
    max_steps: Optional[int] = None,
    goal_radius: Optional[float] = None,
    render_mode: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    validation_start_time = time.perf_counter()

    validation_report = {
        "overall_valid": True,
        "validation_timestamp": time.time(),
        "parameter_results": {},
        "cross_parameter_checks": {},
        "warnings": [],
        "errors": [],
        "recommendations": [],
    }

    try:
        if grid_size is not None:
            grid_valid = True
            grid_issues = []

            if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
                grid_valid = False
                grid_issues.append("Must be tuple/list of exactly 2 elements")
            else:
                width, height = grid_size
                if not isinstance(width, int) or not isinstance(height, int):
                    grid_valid = False
                    grid_issues.append("Dimensions must be integers")
                elif width <= 0 or height <= 0:
                    grid_valid = False
                    grid_issues.append("Dimensions must be positive")
                elif width > 1024 or height > 1024:
                    validation_report["warnings"].append(
                        f"Large grid size ({width}x{height}) may impact performance"
                    )
                elif width < 16 or height < 16:
                    validation_report["warnings"].append(
                        f"Small grid size ({width}x{height}) may limit complexity"
                    )

                if grid_valid:
                    memory_estimate_mb = (width * height * 4) / (1024 * 1024)  # float32
                    if memory_estimate_mb > 100:
                        validation_report["warnings"].append(
                            f"High memory usage: {memory_estimate_mb:.1f}MB"
                        )

            validation_report["parameter_results"]["grid_size"] = {
                "valid": grid_valid,
                "issues": grid_issues,
                "value": grid_size,
            }

            if not grid_valid:
                validation_report["overall_valid"] = False
                validation_report["errors"].extend(grid_issues)

        if source_location is not None:
            source_valid = True
            source_issues = []

            if (
                not isinstance(source_location, (tuple, list))
                or len(source_location) != 2
            ):
                source_valid = False
                source_issues.append("Must be tuple/list of exactly 2 elements")
            else:
                src_x, src_y = source_location
                if not isinstance(src_x, (int, float)) or not isinstance(
                    src_y, (int, float)
                ):
                    source_valid = False
                    source_issues.append("Coordinates must be numeric")
                elif src_x < 0 or src_y < 0:
                    source_valid = False
                    source_issues.append("Coordinates must be non-negative")

                if grid_size is not None and source_valid:
                    width, height = grid_size
                    if src_x >= width or src_y >= height:
                        source_valid = False
                        source_issues.append(
                            f"Coordinates must be within grid bounds: (0,0) to ({width-1},{height-1})"
                        )

            validation_report["parameter_results"]["source_location"] = {
                "valid": source_valid,
                "issues": source_issues,
                "value": source_location,
            }

            if not source_valid:
                validation_report["overall_valid"] = False
                validation_report["errors"].extend(source_issues)

        if max_steps is not None:
            steps_valid = True
            steps_issues = []

            if not isinstance(max_steps, int):
                steps_valid = False
                steps_issues.append("Must be integer")
            elif max_steps <= 0:
                steps_valid = False
                steps_issues.append("Must be positive")
            elif max_steps > 100000:
                validation_report["warnings"].append(
                    f"Very high max_steps ({max_steps}) may impact training"
                )
            elif max_steps < 100:
                validation_report["warnings"].append(
                    f"Low max_steps ({max_steps}) may limit learning"
                )

            validation_report["parameter_results"]["max_steps"] = {
                "valid": steps_valid,
                "issues": steps_issues,
                "value": max_steps,
            }

            if not steps_valid:
                validation_report["overall_valid"] = False
                validation_report["errors"].extend(steps_issues)

        if goal_radius is not None:
            radius_valid = True
            radius_issues = []

            if not isinstance(goal_radius, (int, float)):
                radius_valid = False
                radius_issues.append("Must be numeric")
            elif goal_radius < 0:
                radius_valid = False
                radius_issues.append("Must be non-negative")

            validation_report["parameter_results"]["goal_radius"] = {
                "valid": radius_valid,
                "issues": radius_issues,
                "value": goal_radius,
            }

            if not radius_valid:
                validation_report["overall_valid"] = False
                validation_report["errors"].extend(radius_issues)

        if render_mode is not None:
            render_valid = True
            render_issues = []

            if not isinstance(render_mode, str):
                render_valid = False
                render_issues.append("Must be string")
            elif render_mode not in ["rgb_array", "human"]:
                render_valid = False
                render_issues.append("Must be one of: rgb_array, human")

            validation_report["parameter_results"]["render_mode"] = {
                "valid": render_valid,
                "issues": render_issues,
                "value": render_mode,
            }

            if not render_valid:
                validation_report["overall_valid"] = False
                validation_report["errors"].extend(render_issues)

        if (
            grid_size is not None
            and source_location is not None
            and goal_radius is not None
        ):
            try:
                width, height = grid_size
                src_x, src_y = source_location

                min_distance_to_edge = min(
                    src_x, src_y, width - src_x - 1, height - src_y - 1
                )
                if goal_radius > min_distance_to_edge:
                    validation_report["warnings"].append(
                        f"Goal radius ({goal_radius}) extends beyond grid boundaries from source"
                    )

                validation_report["cross_parameter_checks"][
                    "goal_radius_feasibility"
                ] = {
                    "min_distance_to_edge": min_distance_to_edge,
                    "goal_extends_beyond_grid": goal_radius > min_distance_to_edge,
                }

            except Exception as cross_check_error:
                validation_report["warnings"].append(
                    f"Cross-parameter validation failed: {cross_check_error}"
                )

        validation_time_ms = (time.perf_counter() - validation_start_time) * 1000
        validation_report["validation_time_ms"] = validation_time_ms

        if validation_report["overall_valid"] and not validation_report["warnings"]:
            validation_report["recommendations"].append(
                "All parameters valid and ready for use"
            )
        elif validation_report["warnings"]:
            validation_report["recommendations"].append(
                "Review warnings for potential optimizations"
            )

        if not validation_report["overall_valid"]:
            validation_report["recommendations"].append(
                "Fix validation errors before proceeding"
            )
            validation_report["recommendations"].append(
                "Use default parameters for reliable configuration"
            )

        return validation_report["overall_valid"], validation_report

    except Exception as e:
        validation_time_ms = (time.perf_counter() - validation_start_time) * 1000

        return False, {
            "overall_valid": False,
            "validation_timestamp": time.time(),
            "validation_time_ms": validation_time_ms,
            "error": str(e),
            "errors": [f"Validation process failed: {e}"],
            "recommendations": ["Check parameter types and try again"],
        }


def _log_environment_creation(
    env_type: str,
    config_used: Dict[str, Any],
    creation_time: float,
    creation_success: bool,
) -> None:
    try:
        log_message = (
            f"Environment creation: type={env_type}, success={creation_success}"
        )

        log_message += f", creation_time={creation_time:.2f}ms"

        if creation_success:
            _module_logger.info(log_message)
        else:
            _module_logger.error(log_message)

        config_summary = {
            key: value
            for key, value in config_used.items()
            if key
            in [
                "grid_size",
                "source_location",
                "max_steps",
                "goal_radius",
                "render_mode",
            ]
        }
        _module_logger.debug(f"Configuration used: {config_summary}")

        if creation_time > 100:  # 100ms threshold
            _module_logger.warning(
                f"Environment creation exceeded target time: {creation_time:.2f}ms > 100ms"
            )
        elif creation_time > 50:  # 50ms warning threshold
            _module_logger.info(
                f"Environment creation time above optimal: {creation_time:.2f}ms"
            )

        if "grid_size" in config_used:
            try:
                width, height = config_used["grid_size"]
                memory_estimate_mb = (width * height * 4) / (1024 * 1024)
                _module_logger.debug(
                    f"Estimated memory usage: {memory_estimate_mb:.1f}MB"
                )
            except Exception:
                pass  # Skip memory estimation if calculation fails

    except Exception as log_error:
        _module_logger.warning(f"Creation logging failed: {log_error}")


_module_logger.info(
    f"Plume navigation environment module initialized: version={ENVIRONMENT_VERSION}, "
    f"supported_environments={len(SUPPORTED_ENVIRONMENTS)}, env_id={ENV_ID}"
)

try:
    from ..constants import validate_constant_consistency

    constants_valid, validation_report = validate_constant_consistency(
        strict_mode=False
    )

    if not constants_valid:
        _module_logger.warning(
            f"Default constants validation issues: {validation_report.get('errors', [])}"
        )
    else:
        _module_logger.debug("Default constants validation passed")

except Exception as validation_error:
    _module_logger.warning(f"Module validation check failed: {validation_error}")

_module_logger.debug(f"Environment module ready: exports={len(__all__)} items")
