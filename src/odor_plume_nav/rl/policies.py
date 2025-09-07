"""
Custom policy implementations and network architectures for odor plume navigation tasks.

This module provides specialized neural network designs optimized for multi-modal observation 
processing and continuous control of navigation agents. The custom policy networks can process 
odor concentration, agent position, and orientation observations more effectively than standard 
stable-baselines3 policies, enabling enhanced training performance for olfactory navigation 
research.

Key Features:
- Multi-modal feature extraction networks for heterogeneous sensor data
- Domain-specific preprocessing and normalization strategies
- Configurable network architectures through Hydra configuration schema
- PyTorch-based implementations compatible with stable-baselines3
- Specialized networks for continuous control with speed/angular velocity outputs
- Observation space adaptation for single and multi-sensor configurations

Technical Architecture:
- Feature extraction layers process each observation modality independently
- Fusion networks combine multi-modal representations for policy decisions
- Configurable activation functions, layer sizes, and dropout rates
- Support for both shared and separate value/policy network architectures
- Integration with existing NavigatorProtocol and GymnasiumEnv components
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Type, Union, Any, Callable
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Stable-baselines3 imports for custom policy framework
try:
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.preprocessing import preprocess_obs
    from stable_baselines3.common.type_aliases import Schedule
    from stable_baselines3.common.utils import get_device
    import gym
except ImportError as e:
from loguru import logger
    logger.error("stable-baselines3 dependency is missing: %s", e)
    raise

# Configuration imports
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict

# Local imports - dependencies analyzed from context
from odor_plume_nav.environments.spaces import ObservationType, ActionType
from odor_plume_nav.config.models import resolve_env_value, validate_env_interpolation

# Enhanced logging
try:
    from odor_plume_nav.utils.logging_setup import get_enhanced_logger
    logger = get_enhanced_logger(__name__)
except ImportError:
from loguru import logger
class OdorConcentrationExtractor(nn.Module):
    """
    Specialized feature extractor for odor concentration data.
    
    Processes scalar odor concentration values through a dedicated network
    designed to capture the nuanced relationships in olfactory navigation.
    The network applies domain-specific transformations to enhance the
    representation of concentration gradients and temporal dynamics.
    
    Architecture:
        - Input normalization with adaptive scaling
        - Multi-layer perceptron with skip connections
        - Specialized activation functions for concentration processing
        - Output feature representation for fusion networks
    
    Args:
        input_dim: Dimension of odor concentration input (typically 1)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function type ('relu', 'tanh', 'elu')
        dropout_rate: Dropout probability for regularization
        use_batch_norm: Whether to apply batch normalization
        concentration_scaling: Scaling factor for concentration values
        
    Examples:
        >>> extractor = OdorConcentrationExtractor(
        ...     input_dim=1,
        ...     hidden_dims=[32, 64, 32],
        ...     activation='elu',
        ...     dropout_rate=0.1
        ... )
        >>> concentration = torch.tensor([0.75], dtype=torch.float32)
        >>> features = extractor(concentration)
        >>> print(features.shape)  # torch.Size([32])
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = [32, 64, 32],
        activation: str = 'elu',
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        concentration_scaling: float = 1.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.concentration_scaling = concentration_scaling
        
        # Activation function selection
        self.activation_fn = self._get_activation_function(activation)
        
        # Build feature extraction layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation_fn)
            
            # Dropout (not on last layer)
            if dropout_rate > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        # Initialize weights with specialized initialization for concentration data
        self._initialize_weights()
        
        logger.debug(
            f"OdorConcentrationExtractor initialized: "
            f"input_dim={input_dim}, hidden_dims={hidden_dims}, "
            f"output_dim={self.output_dim}, activation={activation}"
        )
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name with specialized options."""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'swish': nn.SiLU(),  # Swish/SiLU activation
            'gelu': nn.GELU()
        }
        
        if activation not in activation_map:
            logger.warning(f"Unknown activation '{activation}', using ReLU")
            return nn.ReLU()
        
        return activation_map[activation]
    
    def _initialize_weights(self):
        """Initialize weights with specialized scheme for concentration processing."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, concentration: torch.Tensor) -> torch.Tensor:
        """
        Extract features from odor concentration data.
        
        Args:
            concentration: Tensor of shape (batch_size,) or (batch_size, 1)
                containing normalized odor concentration values [0, 1]
            
        Returns:
            torch.Tensor: Feature representation of shape (batch_size, output_dim)
        """
        # Ensure proper input shape
        if concentration.dim() == 1:
            concentration = concentration.unsqueeze(-1)
        
        # Apply concentration scaling if specified
        if self.concentration_scaling != 1.0:
            concentration = concentration * self.concentration_scaling
        
        # Extract features
        features = self.feature_extractor(concentration)
        
        return features


class SpatialPositionExtractor(nn.Module):
    """
    Specialized feature extractor for spatial position and orientation data.
    
    Processes agent position (x, y coordinates) and orientation (angle) through
    network architectures designed to capture spatial relationships and geometric
    constraints in navigation tasks. Includes specialized encoding for periodic
    orientation data and distance-based position normalization.
    
    Architecture:
        - Sinusoidal encoding for periodic orientation data
        - Position normalization based on environment bounds
        - Multi-layer processing with spatial-aware activations
        - Optional coordinate transformation layers
        - Feature fusion for combined spatial representation
    
    Args:
        position_dim: Dimension of position input (typically 2 for x,y)
        orientation_dim: Dimension of orientation input (typically 1)
        hidden_dims: List of hidden layer dimensions for spatial processing
        activation: Activation function type
        use_sinusoidal_encoding: Whether to use sinusoidal encoding for orientation
        environment_bounds: Optional (width, height) for position normalization
        coordinate_frame: Coordinate transformation ('cartesian', 'polar', 'hybrid')
        
    Examples:
        >>> extractor = SpatialPositionExtractor(
        ...     position_dim=2,
        ...     orientation_dim=1,
        ...     hidden_dims=[64, 128, 64],
        ...     use_sinusoidal_encoding=True,
        ...     environment_bounds=(640, 480)
        ... )
        >>> position = torch.tensor([320.0, 240.0], dtype=torch.float32)
        >>> orientation = torch.tensor([45.0], dtype=torch.float32)
        >>> features = extractor(position, orientation)
    """
    
    def __init__(
        self,
        position_dim: int = 2,
        orientation_dim: int = 1,
        hidden_dims: List[int] = [64, 128, 64],
        activation: str = 'relu',
        use_sinusoidal_encoding: bool = True,
        environment_bounds: Optional[Tuple[float, float]] = None,
        coordinate_frame: str = 'cartesian',
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.position_dim = position_dim
        self.orientation_dim = orientation_dim
        self.use_sinusoidal_encoding = use_sinusoidal_encoding
        self.environment_bounds = environment_bounds
        self.coordinate_frame = coordinate_frame
        
        # Calculate input dimensions based on encoding
        self.encoded_position_dim = position_dim
        if coordinate_frame == 'polar':
            self.encoded_position_dim = 2  # radius, angle
        elif coordinate_frame == 'hybrid':
            self.encoded_position_dim = position_dim + 2  # x,y + r,theta
        
        self.encoded_orientation_dim = orientation_dim
        if use_sinusoidal_encoding:
            self.encoded_orientation_dim = orientation_dim * 2  # sin, cos
        
        total_input_dim = self.encoded_position_dim + self.encoded_orientation_dim
        
        # Activation function
        self.activation_fn = self._get_activation_function(activation)
        
        # Build spatial processing network
        layers = []
        prev_dim = total_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation_fn)
            
            if dropout_rate > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.spatial_processor = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else total_input_dim
        
        # Initialize weights
        self._initialize_weights()
        
        logger.debug(
            f"SpatialPositionExtractor initialized: "
            f"position_dim={position_dim}, orientation_dim={orientation_dim}, "
            f"encoded_dims=({self.encoded_position_dim}, {self.encoded_orientation_dim}), "
            f"output_dim={self.output_dim}, coordinate_frame={coordinate_frame}"
        )
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'swish': nn.SiLU(),
            'gelu': nn.GELU()
        }
        return activation_map.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize weights for spatial processing."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def _encode_position(self, position: torch.Tensor) -> torch.Tensor:
        """Encode position based on coordinate frame."""
        if self.coordinate_frame == 'cartesian':
            # Normalize position if bounds provided
            if self.environment_bounds is not None:
                width, height = self.environment_bounds
                normalized_pos = position.clone()
                normalized_pos[:, 0] = normalized_pos[:, 0] / width  # x normalization
                normalized_pos[:, 1] = normalized_pos[:, 1] / height  # y normalization
                return normalized_pos
            return position
        
        elif self.coordinate_frame == 'polar':
            # Convert to polar coordinates
            x, y = position[:, 0], position[:, 1]
            r = torch.sqrt(x**2 + y**2)
            theta = torch.atan2(y, x)
            return torch.stack([r, theta], dim=1)
        
        elif self.coordinate_frame == 'hybrid':
            # Combine cartesian and polar
            cartesian = self._encode_position_cartesian(position)
            polar = self._encode_position_polar(position)
            return torch.cat([cartesian, polar], dim=1)
        
        else:
            raise ValueError(f"Unknown coordinate_frame: {self.coordinate_frame}")
    
    def _encode_position_cartesian(self, position: torch.Tensor) -> torch.Tensor:
        """Helper for cartesian encoding."""
        if self.environment_bounds is not None:
            width, height = self.environment_bounds
            normalized_pos = position.clone()
            normalized_pos[:, 0] = normalized_pos[:, 0] / width
            normalized_pos[:, 1] = normalized_pos[:, 1] / height
            return normalized_pos
        return position
    
    def _encode_position_polar(self, position: torch.Tensor) -> torch.Tensor:
        """Helper for polar encoding."""
        x, y = position[:, 0], position[:, 1]
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        return torch.stack([r, theta], dim=1)
    
    def _encode_orientation(self, orientation: torch.Tensor) -> torch.Tensor:
        """Encode orientation with optional sinusoidal encoding."""
        if self.use_sinusoidal_encoding:
            # Convert degrees to radians and compute sin/cos
            orientation_rad = orientation * (math.pi / 180.0)
            sin_orient = torch.sin(orientation_rad)
            cos_orient = torch.cos(orientation_rad)
            return torch.cat([sin_orient, cos_orient], dim=1)
        else:
            # Normalize orientation to [0, 1] range
            return orientation / 360.0
    
    def forward(
        self, 
        position: torch.Tensor, 
        orientation: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features from spatial position and orientation data.
        
        Args:
            position: Tensor of shape (batch_size, 2) containing [x, y] coordinates
            orientation: Tensor of shape (batch_size, 1) containing orientation in degrees
            
        Returns:
            torch.Tensor: Spatial feature representation of shape (batch_size, output_dim)
        """
        # Ensure proper input shapes
        if position.dim() == 1:
            position = position.unsqueeze(0)
        if orientation.dim() == 1:
            orientation = orientation.unsqueeze(-1)
        
        # Encode position and orientation
        encoded_position = self._encode_position(position)
        encoded_orientation = self._encode_orientation(orientation)
        
        # Concatenate spatial features
        spatial_input = torch.cat([encoded_position, encoded_orientation], dim=1)
        
        # Process through spatial network
        features = self.spatial_processor(spatial_input)
        
        return features


class MultiSensorExtractor(nn.Module):
    """
    Feature extractor for multi-sensor array data in navigation tasks.
    
    Processes arrays of sensor readings from multiple sensors arranged in
    various geometric configurations (bilateral, triangular, custom). The
    network includes specialized processing for sensor correlations and
    spatial relationships between sensor positions.
    
    Architecture:
        - Sensor-wise processing with shared weights
        - Attention mechanisms for sensor importance weighting
        - Spatial correlation modeling between sensors
        - Configurable aggregation strategies (mean, max, attention)
        - Dropout and regularization for robust multi-sensor fusion
    
    Args:
        num_sensors: Number of sensors in the array
        sensor_dim: Dimension of each sensor reading (typically 1)
        hidden_dims: Hidden layer dimensions for sensor processing
        aggregation_method: How to combine sensor features ('attention', 'mean', 'max')
        use_sensor_positions: Whether to include sensor position encoding
        sensor_layout: Sensor arrangement type ('bilateral', 'triangular', 'custom')
        
    Examples:
        >>> extractor = MultiSensorExtractor(
        ...     num_sensors=3,
        ...     sensor_dim=1,
        ...     hidden_dims=[32, 64],
        ...     aggregation_method='attention'
        ... )
        >>> sensor_readings = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32)
        >>> features = extractor(sensor_readings)
    """
    
    def __init__(
        self,
        num_sensors: int = 2,
        sensor_dim: int = 1,
        hidden_dims: List[int] = [32, 64],
        aggregation_method: str = 'attention',
        use_sensor_positions: bool = True,
        sensor_layout: str = 'bilateral',
        activation: str = 'relu',
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.sensor_dim = sensor_dim
        self.aggregation_method = aggregation_method
        self.use_sensor_positions = use_sensor_positions
        self.sensor_layout = sensor_layout
        
        # Calculate input dimension including optional position encoding
        input_dim = sensor_dim
        if use_sensor_positions:
            input_dim += 2  # x, y position encoding for each sensor
        
        # Activation function
        self.activation_fn = self._get_activation_function(activation)
        
        # Sensor-wise feature extraction (shared across sensors)
        sensor_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            sensor_layers.append(nn.Linear(prev_dim, hidden_dim))
            sensor_layers.append(self.activation_fn)
            if dropout_rate > 0:
                sensor_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.sensor_processor = nn.Sequential(*sensor_layers)
        self.sensor_feature_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        # Aggregation mechanism
        if aggregation_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.sensor_feature_dim, self.sensor_feature_dim // 2),
                self.activation_fn,
                nn.Linear(self.sensor_feature_dim // 2, 1)
            )
        
        # Output dimension
        self.output_dim = self.sensor_feature_dim
        
        # Sensor position encoding if used
        if use_sensor_positions:
            self.sensor_positions = self._generate_sensor_positions()
        
        self._initialize_weights()
        
        logger.debug(
            f"MultiSensorExtractor initialized: "
            f"num_sensors={num_sensors}, sensor_dim={sensor_dim}, "
            f"output_dim={self.output_dim}, aggregation={aggregation_method}, "
            f"layout={sensor_layout}"
        )
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'swish': nn.SiLU(),
            'gelu': nn.GELU()
        }
        return activation_map.get(activation, nn.ReLU())
    
    def _generate_sensor_positions(self) -> torch.Tensor:
        """Generate sensor position encodings based on layout."""
        if self.sensor_layout == 'bilateral':
            # Two sensors positioned left and right
            positions = torch.tensor([
                [-1.0, 0.0],  # Left sensor
                [1.0, 0.0]    # Right sensor
            ], dtype=torch.float32)
            
        elif self.sensor_layout == 'triangular':
            # Three sensors in triangular arrangement
            positions = torch.tensor([
                [0.0, 1.0],     # Front sensor
                [-0.866, -0.5], # Left-back sensor
                [0.866, -0.5]   # Right-back sensor
            ], dtype=torch.float32)
            
        else:  # custom or other layouts
            # Uniform circular arrangement
            angles = torch.linspace(0, 2 * math.pi, self.num_sensors + 1)[:-1]
            positions = torch.stack([
                torch.cos(angles),
                torch.sin(angles)
            ], dim=1)
        
        # Ensure correct number of sensors
        if positions.shape[0] != self.num_sensors:
            # Interpolate or repeat to match num_sensors
            if self.num_sensors == 1:
                positions = torch.zeros(1, 2)
            else:
                # Use circular arrangement as fallback
                angles = torch.linspace(0, 2 * math.pi, self.num_sensors + 1)[:-1]
                positions = torch.stack([
                    torch.cos(angles),
                    torch.sin(angles)
                ], dim=1)
        
        return positions
    
    def _initialize_weights(self):
        """Initialize weights for multi-sensor processing."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, sensor_readings: torch.Tensor) -> torch.Tensor:
        """
        Extract features from multi-sensor array data.
        
        Args:
            sensor_readings: Tensor of shape (batch_size, num_sensors) containing
                sensor values, or (batch_size,) if single sensor
            
        Returns:
            torch.Tensor: Multi-sensor feature representation of shape (batch_size, output_dim)
        """
        batch_size = sensor_readings.shape[0]
        
        # Handle single sensor case
        if sensor_readings.dim() == 1:
            sensor_readings = sensor_readings.unsqueeze(0)
        if sensor_readings.shape[1] != self.num_sensors:
            # Pad or truncate to match expected number of sensors
            if sensor_readings.shape[1] < self.num_sensors:
                padding = torch.zeros(batch_size, self.num_sensors - sensor_readings.shape[1])
                sensor_readings = torch.cat([sensor_readings, padding], dim=1)
            else:
                sensor_readings = sensor_readings[:, :self.num_sensors]
        
        # Prepare input for each sensor
        sensor_features = []
        
        for i in range(self.num_sensors):
            sensor_input = sensor_readings[:, i:i+1]  # Shape: (batch_size, 1)
            
            # Add position encoding if enabled
            if self.use_sensor_positions:
                pos_encoding = self.sensor_positions[i:i+1].expand(batch_size, -1)
                sensor_input = torch.cat([sensor_input, pos_encoding], dim=1)
            
            # Process through sensor network
            sensor_feat = self.sensor_processor(sensor_input)
            sensor_features.append(sensor_feat)
        
        # Stack sensor features: (batch_size, num_sensors, feature_dim)
        sensor_features = torch.stack(sensor_features, dim=1)
        
        # Aggregate features across sensors
        if self.aggregation_method == 'mean':
            aggregated = torch.mean(sensor_features, dim=1)
        elif self.aggregation_method == 'max':
            aggregated, _ = torch.max(sensor_features, dim=1)
        elif self.aggregation_method == 'attention':
            # Compute attention weights
            attention_scores = self.attention(sensor_features)  # (batch_size, num_sensors, 1)
            attention_weights = torch.softmax(attention_scores, dim=1)
            
            # Apply attention weights
            aggregated = torch.sum(sensor_features * attention_weights, dim=1)
        else:
            # Default to mean aggregation
            aggregated = torch.mean(sensor_features, dim=1)
        
        return aggregated


class PlumeNavigationFeaturesExtractor(BaseFeaturesExtractor):
    """
    Multi-modal feature extractor for odor plume navigation environments.
    
    This is the main feature extractor that combines specialized processing
    for different observation modalities in the plume navigation environment.
    It implements the stable-baselines3 BaseFeaturesExtractor interface while
    providing domain-specific multi-modal processing capabilities.
    
    The extractor processes the gymnasium.spaces.Dict observation space containing:
    - odor_concentration: Scalar odor intensity [0, 1]
    - agent_position: Agent coordinates [x, y] 
    - agent_orientation: Agent heading in degrees [0, 360)
    - multi_sensor_readings: Optional array of additional sensor values
    
    Architecture:
        - Specialized sub-extractors for each observation modality
        - Feature fusion network combining multi-modal representations
        - Configurable fusion strategies (concatenation, attention, gating)
        - Output normalization and dimensionality reduction
        - Dropout and regularization for robust training
    
    Args:
        observation_space: gymnasium.spaces.Dict defining observation structure
        features_dim: Output feature dimension (default: 256)
        config: Optional configuration dictionary for network parameters
        
    Examples:
        >>> from gymnasium.spaces import Dict, Box
        >>> obs_space = Dict({
        ...     'odor_concentration': Box(low=0.0, high=1.0, shape=()),
        ...     'agent_position': Box(low=0.0, high=640.0, shape=(2,)),
        ...     'agent_orientation': Box(low=0.0, high=360.0, shape=()),
        ...     'multi_sensor_readings': Box(low=0.0, high=1.0, shape=(3,))
        ... })
        >>> extractor = PlumeNavigationFeaturesExtractor(obs_space, features_dim=128)
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        config: Optional[Dict[str, Any]] = None
    ):
        # Initialize base class
        super().__init__(observation_space, features_dim)
        
        # Parse configuration
        self.config = config or {}
        
        # Extract observation space components
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("PlumeNavigationFeaturesExtractor requires Dict observation space")
        
        self.obs_space = observation_space.spaces
        
        # Validate required observation components
        required_keys = {'odor_concentration', 'agent_position', 'agent_orientation'}
        if not required_keys.issubset(self.obs_space.keys()):
            raise ValueError(f"Observation space must contain keys: {required_keys}")
        
        # Check for optional multi-sensor component
        self.has_multi_sensor = 'multi_sensor_readings' in self.obs_space
        
        # Configuration parameters with defaults
        odor_config = self.config.get('odor_extractor', {})
        spatial_config = self.config.get('spatial_extractor', {})
        sensor_config = self.config.get('sensor_extractor', {})
        fusion_config = self.config.get('fusion', {})
        
        # Initialize odor concentration extractor
        self.odor_extractor = OdorConcentrationExtractor(
            input_dim=1,
            hidden_dims=odor_config.get('hidden_dims', [32, 64, 32]),
            activation=odor_config.get('activation', 'elu'),
            dropout_rate=odor_config.get('dropout_rate', 0.1),
            concentration_scaling=odor_config.get('concentration_scaling', 1.0)
        )
        
        # Extract environment bounds from position space
        env_bounds = None
        if hasattr(self.obs_space['agent_position'], 'high'):
            pos_high = self.obs_space['agent_position'].high
            if len(pos_high) >= 2:
                env_bounds = (float(pos_high[0]), float(pos_high[1]))
        
        # Initialize spatial position extractor
        self.spatial_extractor = SpatialPositionExtractor(
            position_dim=2,
            orientation_dim=1,
            hidden_dims=spatial_config.get('hidden_dims', [64, 128, 64]),
            activation=spatial_config.get('activation', 'relu'),
            use_sinusoidal_encoding=spatial_config.get('use_sinusoidal_encoding', True),
            environment_bounds=env_bounds,
            coordinate_frame=spatial_config.get('coordinate_frame', 'cartesian'),
            dropout_rate=spatial_config.get('dropout_rate', 0.1)
        )
        
        # Initialize multi-sensor extractor if needed
        if self.has_multi_sensor:
            sensor_shape = self.obs_space['multi_sensor_readings'].shape
            num_sensors = sensor_shape[0] if len(sensor_shape) > 0 else 1
            
            self.sensor_extractor = MultiSensorExtractor(
                num_sensors=num_sensors,
                sensor_dim=1,
                hidden_dims=sensor_config.get('hidden_dims', [32, 64]),
                aggregation_method=sensor_config.get('aggregation_method', 'attention'),
                use_sensor_positions=sensor_config.get('use_sensor_positions', True),
                sensor_layout=sensor_config.get('sensor_layout', 'bilateral'),
                dropout_rate=sensor_config.get('dropout_rate', 0.1)
            )
        else:
            self.sensor_extractor = None
        
        # Calculate total feature dimension from extractors
        total_extractor_dim = (
            self.odor_extractor.output_dim + 
            self.spatial_extractor.output_dim +
            (self.sensor_extractor.output_dim if self.sensor_extractor else 0)
        )
        
        # Feature fusion network
        fusion_hidden_dims = fusion_config.get('hidden_dims', [512, 256])
        fusion_activation = fusion_config.get('activation', 'relu')
        fusion_dropout = fusion_config.get('dropout_rate', 0.2)
        
        fusion_layers = []
        prev_dim = total_extractor_dim
        
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(prev_dim, hidden_dim))
            fusion_layers.append(self._get_activation_function(fusion_activation))
            if fusion_dropout > 0:
                fusion_layers.append(nn.Dropout(fusion_dropout))
            prev_dim = hidden_dim
        
        # Final output layer
        fusion_layers.append(nn.Linear(prev_dim, features_dim))
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(
            f"PlumeNavigationFeaturesExtractor initialized: "
            f"features_dim={features_dim}, has_multi_sensor={self.has_multi_sensor}, "
            f"total_extractor_dim={total_extractor_dim}"
        )
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'swish': nn.SiLU(),
            'gelu': nn.GELU()
        }
        return activation_map.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize fusion network weights."""
        for module in self.fusion_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract and fuse features from multi-modal observations.
        
        Args:
            observations: Dictionary containing observation tensors:
                - odor_concentration: (batch_size,) or (batch_size, 1)
                - agent_position: (batch_size, 2) 
                - agent_orientation: (batch_size,) or (batch_size, 1)
                - multi_sensor_readings: (batch_size, num_sensors) [optional]
                
        Returns:
            torch.Tensor: Fused feature representation of shape (batch_size, features_dim)
        """
        # Extract features from each modality
        feature_components = []
        
        # Process odor concentration
        odor_features = self.odor_extractor(observations['odor_concentration'])
        feature_components.append(odor_features)
        
        # Process spatial information
        spatial_features = self.spatial_extractor(
            observations['agent_position'],
            observations['agent_orientation']
        )
        feature_components.append(spatial_features)
        
        # Process multi-sensor data if available
        if self.has_multi_sensor and self.sensor_extractor is not None:
            sensor_features = self.sensor_extractor(observations['multi_sensor_readings'])
            feature_components.append(sensor_features)
        
        # Concatenate all feature components
        combined_features = torch.cat(feature_components, dim=1)
        
        # Apply fusion network
        fused_features = self.fusion_network(combined_features)
        
        return fused_features


class PlumeNavigationPolicy(ActorCriticPolicy):
    """
    Custom actor-critic policy for odor plume navigation tasks.
    
    This policy extends stable-baselines3's ActorCriticPolicy with domain-specific
    components optimized for olfactory navigation. It uses the specialized
    PlumeNavigationFeaturesExtractor for multi-modal observation processing
    and implements continuous control for speed and angular velocity outputs.
    
    Key Features:
    - Multi-modal observation processing through custom feature extractor
    - Continuous action space for navigation control (speed, angular_velocity)
    - Configurable network architectures for actor and critic
    - Domain-specific initialization and regularization strategies
    - Support for both shared and separate actor-critic feature extraction
    
    Args:
        observation_space: Gymnasium observation space (Dict)
        action_space: Gymnasium action space (Box with shape (2,))
        lr_schedule: Learning rate schedule
        net_arch: Network architecture specification
        activation_fn: Activation function for policy networks
        features_extractor_class: Custom features extractor class
        features_extractor_kwargs: Arguments for features extractor
        
    Examples:
        >>> from stable_baselines3 import PPO
        >>> env = GymnasiumEnv(...)
        >>> policy_kwargs = {
        ...     'features_extractor_class': PlumeNavigationPolicy,
        ...     'features_extractor_kwargs': {
        ...         'features_dim': 256,
        ...         'config': {'fusion': {'dropout_rate': 0.1}}
        ...     }
        ... }
        >>> model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs)
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_extractor_class: Type[BaseFeaturesExtractor] = PlumeNavigationFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        # Store configuration
        self.config = config or {}
        
        # Set default network architecture if not provided
        if net_arch is None:
            net_arch = [
                {'pi': [256, 128], 'vf': [256, 128]}  # Separate networks for policy and value
            ]
        
        # Set default features extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {
                'features_dim': 256,
                'config': self.config.get('features_extractor', {})
            }
        
        # Initialize parent class
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs or {}
        )
        
        # Validate action space for navigation tasks
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError("PlumeNavigationPolicy requires Box action space")
        
        if action_space.shape != (2,):
            raise ValueError("Action space must have shape (2,) for [speed, angular_velocity]")
        
        # Domain-specific initialization
        self._apply_domain_specific_initialization()
        
        logger.info(
            f"PlumeNavigationPolicy initialized: "
            f"obs_space={type(observation_space).__name__}, "
            f"action_space={action_space.shape}, "
            f"net_arch={net_arch}"
        )
    
    def _apply_domain_specific_initialization(self):
        """Apply specialized weight initialization for navigation tasks."""
        # Custom initialization for action distribution
        # Initialize action network with smaller weights for more stable initial exploration
        if hasattr(self, 'action_net'):
            for module in self.action_net.modules():
                if isinstance(module, nn.Linear):
                    # Smaller initialization for action outputs
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        # Initialize log_std with conservative values for navigation control
        if hasattr(self, 'log_std'):
            # Initialize with log(0.3) for moderate exploration
            nn.init.constant_(self.log_std, math.log(0.3))
    
    def _build(self, lr_schedule: Schedule) -> None:
        """Build the policy and value networks."""
        # Build parent networks
        super()._build(lr_schedule)
        
        # Apply domain-specific initialization after building
        self._apply_domain_specific_initialization()
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict state values for given observations.
        
        Args:
            obs: Observations tensor
            
        Returns:
            torch.Tensor: Predicted state values
        """
        features = self.extract_features(obs)
        return self.value_net(features)
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations.
        
        Args:
            obs: Observations tensor
            actions: Actions tensor
            
        Returns:
            Tuple of (values, log_prob, entropy)
        """
        features = self.extract_features(obs)
        distribution = self._get_action_dist_from_latent(features)
        
        log_prob = distribution.log_prob(actions)
        values = self.value_net(features)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy


def create_policy_config(
    features_dim: int = 256,
    odor_extractor_config: Optional[Dict[str, Any]] = None,
    spatial_extractor_config: Optional[Dict[str, Any]] = None,
    sensor_extractor_config: Optional[Dict[str, Any]] = None,
    fusion_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a configuration dictionary for PlumeNavigationPolicy.
    
    This function provides a convenient way to create policy configurations
    with sensible defaults for odor plume navigation tasks while allowing
    customization of individual components.
    
    Args:
        features_dim: Output dimension of the features extractor
        odor_extractor_config: Configuration for odor concentration processing
        spatial_extractor_config: Configuration for spatial position processing
        sensor_extractor_config: Configuration for multi-sensor processing
        fusion_config: Configuration for feature fusion network
        **kwargs: Additional configuration parameters
        
    Returns:
        Dict[str, Any]: Complete policy configuration dictionary
        
    Examples:
        >>> config = create_policy_config(
        ...     features_dim=128,
        ...     fusion_config={'dropout_rate': 0.1},
        ...     odor_extractor_config={'activation': 'elu'}
        ... )
        >>> policy_kwargs = {
        ...     'features_extractor_class': PlumeNavigationFeaturesExtractor,
        ...     'features_extractor_kwargs': {
        ...         'features_dim': config['features_dim'],
        ...         'config': config
        ...     }
        ... }
    """
    # Default configurations for each component
    default_odor_config = {
        'hidden_dims': [32, 64, 32],
        'activation': 'elu',
        'dropout_rate': 0.1,
        'concentration_scaling': 1.0
    }
    
    default_spatial_config = {
        'hidden_dims': [64, 128, 64],
        'activation': 'relu',
        'use_sinusoidal_encoding': True,
        'coordinate_frame': 'cartesian',
        'dropout_rate': 0.1
    }
    
    default_sensor_config = {
        'hidden_dims': [32, 64],
        'aggregation_method': 'attention',
        'use_sensor_positions': True,
        'sensor_layout': 'bilateral',
        'dropout_rate': 0.1
    }
    
    default_fusion_config = {
        'hidden_dims': [512, 256],
        'activation': 'relu',
        'dropout_rate': 0.2
    }
    
    # Merge with provided configurations
    config = {
        'features_dim': features_dim,
        'odor_extractor': {**default_odor_config, **(odor_extractor_config or {})},
        'spatial_extractor': {**default_spatial_config, **(spatial_extractor_config or {})},
        'sensor_extractor': {**default_sensor_config, **(sensor_extractor_config or {})},
        'fusion': {**default_fusion_config, **(fusion_config or {})}
    }
    
    # Add any additional kwargs
    config.update(kwargs)
    
    return config


def create_policy_from_config(config: Union[Dict[str, Any], DictConfig]) -> Dict[str, Any]:
    """
    Create stable-baselines3 policy_kwargs from configuration.
    
    This function converts a configuration dictionary (potentially from Hydra)
    into the format expected by stable-baselines3 algorithms, handling the
    mapping between configuration parameters and policy constructor arguments.
    
    Args:
        config: Configuration dictionary or DictConfig containing policy parameters
        
    Returns:
        Dict[str, Any]: policy_kwargs suitable for stable-baselines3 algorithms
        
    Examples:
        >>> # From Python dict
        >>> config = {
        ...     'features_dim': 256,
        ...     'net_arch': [{'pi': [256, 128], 'vf': [256, 128]}],
        ...     'activation_fn': 'tanh'
        ... }
        >>> policy_kwargs = create_policy_from_config(config)
        >>> model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs)
        
        >>> # From Hydra config
        >>> @hydra.main(config_path="conf", config_name="rl_config")
        >>> def train(cfg: DictConfig):
        ...     policy_kwargs = create_policy_from_config(cfg.policy)
        ...     model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs)
    """
    # Handle DictConfig conversion
    if HYDRA_AVAILABLE and hasattr(config, 'to_container'):
        config = OmegaConf.to_container(config, resolve=True)
    
    # Extract policy parameters
    features_dim = config.get('features_dim', 256)
    net_arch = config.get('net_arch', [{'pi': [256, 128], 'vf': [256, 128]}])
    
    # Handle activation function specification
    activation_fn_name = config.get('activation_fn', 'tanh')
    activation_fn_map = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU,
        'swish': nn.SiLU,
        'gelu': nn.GELU
    }
    activation_fn = activation_fn_map.get(activation_fn_name, nn.Tanh)
    
    # Create features extractor configuration
    features_extractor_config = {
        'features_dim': features_dim,
        'config': {
            'odor_extractor': config.get('odor_extractor', {}),
            'spatial_extractor': config.get('spatial_extractor', {}),
            'sensor_extractor': config.get('sensor_extractor', {}),
            'fusion': config.get('fusion', {})
        }
    }
    
    # Build policy_kwargs
    policy_kwargs = {
        'features_extractor_class': PlumeNavigationFeaturesExtractor,
        'features_extractor_kwargs': features_extractor_config,
        'net_arch': net_arch,
        'activation_fn': activation_fn
    }
    
    # Add optional parameters if specified
    if 'optimizer_class' in config:
        optimizer_map = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop
        }
        optimizer_class = optimizer_map.get(config['optimizer_class'].lower(), torch.optim.Adam)
        policy_kwargs['optimizer_class'] = optimizer_class
    
    if 'optimizer_kwargs' in config:
        policy_kwargs['optimizer_kwargs'] = config['optimizer_kwargs']
    
    return policy_kwargs


# Utility functions for policy validation and testing

def validate_policy_compatibility(
    observation_space: gym.Space,
    action_space: gym.Space,
    policy_config: Dict[str, Any]
) -> bool:
    """
    Validate that a policy configuration is compatible with given spaces.
    
    Args:
        observation_space: Gymnasium observation space
        action_space: Gymnasium action space  
        policy_config: Policy configuration dictionary
        
    Returns:
        bool: True if configuration is compatible, False otherwise
        
    Examples:
        >>> obs_space = Dict({'odor_concentration': Box(0, 1, shape=())})
        >>> action_space = Box(-1, 1, shape=(2,))
        >>> config = create_policy_config()
        >>> is_valid = validate_policy_compatibility(obs_space, action_space, config)
    """
    try:
        # Check observation space type
        if not isinstance(observation_space, gym.spaces.Dict):
            logger.error("Observation space must be Dict type")
            return False
        
        # Check required observation keys
        required_keys = {'odor_concentration', 'agent_position', 'agent_orientation'}
        if not required_keys.issubset(observation_space.spaces.keys()):
            logger.error(f"Missing required observation keys: {required_keys}")
            return False
        
        # Check action space
        if not isinstance(action_space, gym.spaces.Box):
            logger.error("Action space must be Box type")
            return False
        
        if action_space.shape != (2,):
            logger.error("Action space must have shape (2,)")
            return False
        
        # Validate policy configuration
        if 'features_dim' not in policy_config:
            logger.error("Policy config must specify features_dim")
            return False
        
        features_dim = policy_config['features_dim']
        if not isinstance(features_dim, int) or features_dim <= 0:
            logger.error("features_dim must be positive integer")
            return False
        
        logger.info("Policy configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Policy validation failed: {e}")
        return False


def test_policy_forward_pass(
    observation_space: gym.Space,
    action_space: gym.Space,
    policy_config: Dict[str, Any],
    batch_size: int = 4
) -> bool:
    """
    Test that a policy configuration can perform forward passes correctly.
    
    Args:
        observation_space: Gymnasium observation space
        action_space: Gymnasium action space
        policy_config: Policy configuration dictionary
        batch_size: Batch size for testing
        
    Returns:
        bool: True if forward pass succeeds, False otherwise
        
    Examples:
        >>> obs_space = create_test_observation_space()
        >>> action_space = create_test_action_space()
        >>> config = create_policy_config()
        >>> success = test_policy_forward_pass(obs_space, action_space, config)
    """
    try:
        # Create features extractor
        features_extractor = PlumeNavigationFeaturesExtractor(
            observation_space=observation_space,
            features_dim=policy_config['features_dim'],
            config=policy_config
        )
        
        # Generate sample observations
        sample_obs = {}
        for key, space in observation_space.spaces.items():
            if key == 'odor_concentration':
                sample_obs[key] = torch.rand(batch_size)
            elif key == 'agent_position':
                sample_obs[key] = torch.rand(batch_size, 2) * torch.tensor(space.high)
            elif key == 'agent_orientation':
                sample_obs[key] = torch.rand(batch_size) * 360.0
            elif key == 'multi_sensor_readings':
                sample_obs[key] = torch.rand(batch_size, space.shape[0])
        
        # Test forward pass
        features = features_extractor(sample_obs)
        
        # Validate output shape
        expected_shape = (batch_size, policy_config['features_dim'])
        if features.shape != expected_shape:
            logger.error(f"Expected output shape {expected_shape}, got {features.shape}")
            return False
        
        # Check for NaN or infinite values
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.error("Forward pass produced NaN or infinite values")
            return False
        
        logger.info("Policy forward pass test passed")
        return True
        
    except Exception as e:
        logger.error(f"Forward pass test failed: {e}")
        return False


# Export public API
__all__ = [
    "OdorConcentrationExtractor",
    "SpatialPositionExtractor", 
    "MultiSensorExtractor",
    "PlumeNavigationFeaturesExtractor",
    "PlumeNavigationPolicy",
    "create_policy_config",
    "create_policy_from_config",
    "validate_policy_compatibility",
    "test_policy_forward_pass"
]