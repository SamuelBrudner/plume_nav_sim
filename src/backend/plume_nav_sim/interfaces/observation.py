"""
Observation Model Protocol Definition.

Contract: src/backend/contracts/observation_model_interface.md

This protocol defines the universal interface for observation models, enabling
diverse sensor configurations without environment modification.
"""

from typing import Any, Dict, Protocol, Union, runtime_checkable

import numpy as np

import gymnasium as gym

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    import numpy as np

    NDArray = np.ndarray  # type: ignore[assignment]

# Type alias for observation outputs
ObservationType = Union[NDArray[np.floating], Dict[str, Any], tuple[Any, ...]]


@runtime_checkable
class ObservationModel(Protocol):
    """Protocol defining observation model interface.

    Contract: observation_model_interface.md

    Universal Properties:
        1. Space Containment: observation always in observation_space
        2. Determinism: Same env_state → same observation
        3. Purity: No side effects, no mutations
        4. Shape Consistency: Observation shape matches space

    Type Signature:
        ObservationModel: (EnvironmentState) → ObservationSpace × Observation

    This abstraction is GENERAL - not specific to olfactory sensing.
    Sensors can observe any aspect of environment state: odor, wind,
    obstacles, time, or custom fields added by users.
    """

    @property
    def observation_space(self) -> gym.Space:
        """Gymnasium observation space definition.

        Can be any Gymnasium space type:
        - Box: Vector/tensor observations (e.g., concentration, wind)
        - Dict: Named multi-modal observations (composition)
        - Tuple: Ordered multi-modal observations
        - Discrete: Categorical observations
        - MultiDiscrete: Multiple categorical observations

        Returns:
            Gymnasium Space defining valid observations

        Postconditions:
            C1: Returns valid gym.Space instance
            C2: Space is immutable (same instance every call)
            C3: Space fully defines valid observations
        """
        ...

    def get_observation(self, env_state: Dict[str, Any]) -> ObservationType:
        """Compute observation from environment state.

        Args:
            env_state: Environment state dictionary containing:
                Required:
                - 'agent_state': AgentState (position, orientation, etc.)

                Common (depends on sensor needs):
                - 'plume_field': ConcentrationField (for olfactory sensors)
                - 'wind_field': WindField (for mechanosensory)
                - 'obstacle_map': np.ndarray (for vision/proximity)
                - 'time_step': int (for temporal context)

                Custom fields can be added by users for novel sensors.

        Returns:
            Observation matching observation_space.
            Type depends on space: np.ndarray (Box), dict (Dict), tuple (Tuple)

        Preconditions:
            P1: env_state contains required keys for this sensor
            P2: env_state['agent_state'] is valid AgentState
            P3: Agent position is within environment bounds

        Postconditions:
            C1: observation ∈ self.observation_space
            C2: observation matches space structure (shape, dtype, keys)
            C3: Result is deterministic (same env_state → same observation)

        Properties:
            1. Determinism: Same env_state → same observation
            2. Purity: No side effects, no mutations
            3. Validity: observation_space.contains(observation) = True
        """
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """Return observation model metadata.

        Returns:
            Dictionary containing:
            - 'type': str - Observation model type
            - 'modality': str - Sensory modality (olfactory, visual, etc.)
            - 'parameters': dict - Configuration
            - 'required_state_keys': List[str] - Required env_state keys
        """
        ...
