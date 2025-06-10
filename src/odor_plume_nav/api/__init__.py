1: 1: 1: 1: 1: 1: 1: 1: 1: 1: 1: """
2: 2: 2: 2: 2: 2: 2: 2: 2: 2: 2: Public API module for odor plume navigation library.
3: 3: 3: 3: 3: 3: 3: 3: 3: 3: 3: 
4: 4: 4: 4: 4: 4: 4: 4: 4: 4: 4: This module serves as the primary entry point for external consumers of the odor plume 
5: 5: 5: 5: 5: 5: 5: 5: 5: 5: 5: navigation library, providing unified access to all high-level functionality through 
6: 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: clean, stable interfaces. Re-exports key factory methods, simulation execution functions, 
7: 7: 7: 7: 7: 7: 7: 7: 7: 7: 7: visualization capabilities, and reinforcement learning framework integration to enable 
seamless integration with Kedro pipelines, 
8: 8: 8: 8: 8: 8: 8: 8: 8: 8: 8: stable-baselines3 algorithms, and machine learning analysis tools.
9: 9: 9: 9: 9: 9: 9: 9: 9: 9: 9: 
10: 10: 10: 10: 10: 10: 10: 10: 10: 10: 10: The API is designed with Hydra-based configuration management at its core, supporting
11: 11: 11: 11: 11: 11: 11: 11: 11: 11: 11: sophisticated parameter composition, environment variable interpolation, and 
12: 12: 12: 12: 12: 12: 12: 12: 12: 12: 12: multi-run experiment execution. All functions accept both direct parameters and 
13: 13: 13: 13: 13: 13: 13: 13: 13: 13: 13: Hydra DictConfig objects, ensuring compatibility with diverse research workflows.
14: 14: 14: 14: 14: 14: 14: 14: 14: 14: 14: 
15: 15: 15: 15: 15: 15: 15: 15: 15: 15: 15: Key Features:
16: 16: 16: 16: 16: 16: 16: 16: 16: 16: 16:     - Unified factory methods for navigator and environment creation
17: 17:     - Gymnasium-compliant RL environment interface for modern ML frameworks
18: 18: 17: 17: 17: 17: 17: 17: 17: 17: 17:     - Hydra-based configuration management with hierarchical parameter composition
19: 19: 18: 18: 18: 18: 18: 18: 18: 18: 18:     - Support for both single-agent and multi-agent navigation scenarios  
20: 20: 19: 19: 19: 19: 19: 19: 19: 19: 19:     - Integration with scientific Python ecosystem (NumPy, Matplotlib, OpenCV)
21: 21: 20: 20: 20: 20: 20: 20: 20: 20: 20:     - Protocol-based interfaces ensuring extensibility and algorithm compatibility
22: 22: 21: 21: 21: 21: 21: 21: 21: 21: 21:     - Publication-quality visualization with real-time animation capabilities
23:     - Legacy API migration support for smooth workflow transitions
24: 23: 22: 22: 22: 22: 22: 22: 22: 22: 22: 
25: 24: 23: 23: 23: 23: 23: 23: 23: 23: 23: Supported Import Patterns:
26: 25: 24: 24: 24: 24: 24: 24: 24: 24: 24:     Kedro pipeline integration:
27: 26: 25: 25: 25: 25: 25: 25: 25: 25: 25:         >>> from odor_plume_nav.api import create_navigator, create_video_plume
28: 27: 26: 26: 26: 26: 26: 26: 26: 26: 26:         >>> from odor_plume_nav.api import run_plume_simulation
29: 28: 27: 27: 27: 27: 27: 27: 27: 27: 27:     
30: 29: 28: 28: 28: 28: 28: 28: 28: 28: 28:     Reinforcement learning frameworks:
31: 30: 29: 29: 29: 29: 29: 29: 29: 29: 29:         >>> from odor_plume_nav.api import create_navigator
32: 31: 30: 30: 30: 30: 30: 30: 30: 30: 30:         >>> from odor_plume_nav.core import NavigatorProtocol
33: 32: 31: 31: 31: 31: 31: 31: 31:         >>> import stable_baselines3 as sb3
34: 33: 32: 32: 32: 32: 32: 32: 32:         >>> 
35: 34: 33: 33: 33: 33: 33: 33: 33:         >>> # Create Gymnasium environment for RL training
36: 35: 34: 34: 34: 34: 34: 34: 34:         >>> env = create_gymnasium_environment(cfg.gymnasium)
37: 36: 35: 35: 35: 35: 35: 35: 35:         >>> model = sb3.PPO("MlpPolicy", env, verbose=1)
38: 37: 36: 36: 36: 36: 36: 36: 36:         >>> model.learn(total_timesteps=10000)
39: 38: 37: 37: 37: 37: 37: 37: 37:     
40: 39: 38: 38: 38: 38: 38: 38: 38:     Legacy API migration:
41: 40: 39: 39: 39: 39: 39: 39: 39:         >>> from odor_plume_nav.api import from_legacy
42: 41: 40: 40: 40: 40: 40: 40: 40:         >>> from odor_plume_nav.api import create_navigator, run_plume_simulation
43: 42: 41: 41: 41: 41: 41: 41: 41:         >>> 
44: 43: 42: 42: 42: 42: 42: 42: 42:         >>> # Migrate existing simulation to Gymnasium environment
45: 44: 43: 43: 43: 43: 43: 43: 43:         >>> navigator = create_navigator(cfg.navigator)
46: 45: 44: 44: 44: 44: 44: 44: 44:         >>> plume = create_video_plume(cfg.video_plume)
47: 46: 45: 45: 45: 45: 45: 45: 45:         >>> env = from_legacy(navigator, plume, cfg.simulation)
48: 47: 46: 46: 46: 46: 46: 46: 46: 31: 31:     
49: 48: 47: 47: 47: 47: 47: 47: 47: 32: 32:     Machine learning analysis tools:
50: 49: 48: 48: 48: 48: 48: 48: 48: 33: 33:         >>> from odor_plume_nav.api import create_video_plume, visualize_simulation_results
51: 50: 49: 49: 49: 49: 49: 49: 49: 34: 34:         >>> from odor_plume_nav.utils import set_global_seed
52: 51: 50: 50: 50: 50: 50: 50: 50: 35: 35: 
53: 52: 51: 51: 51: 51: 51: 51: 51: 36: 36: Configuration Management:
54: 53: 52: 52: 52: 52: 52: 52: 52: 37: 37:     All API functions support Hydra-based configuration through DictConfig objects:
55: 54: 53: 53: 53: 53: 53: 53: 53: 38: 38:         >>> from hydra import compose, initialize
56: 55: 54: 54: 54: 54: 54: 54: 54: 39: 39:         >>> from odor_plume_nav.api import create_navigator, create_gymnasium_environment
57: 56: 55: 55: 55: 55: 55: 55: 55: 40: 40:         >>> 
58: 57: 56: 56: 56: 56: 56: 56: 56: 41: 41:         >>> with initialize(config_path="../conf"):
59: 58: 57: 57: 57: 57: 57: 57: 57: 42: 42:         ...     cfg = compose(config_name="config")
60: 59: 58: 58: 58: 58: 58: 58: 58: 43: 43:         ...     navigator = create_navigator(cfg.navigator)
61: 60: 59: 59: 59: 59: 59: 59: 59: 44: 44:         ...     plume = create_video_plume(cfg.video_plume)
62: 61: 60: 60: 60: 60: 60: 60: 60: 45: 45:         ...     results = run_plume_simulation(navigator, plume, cfg.simulation)
63: 62: 61: 61: 61: 61: 61:         ...     
64: 63: 62: 62: 62: 62: 62:         ...     # Or create RL environment
65: 64: 63: 63: 63: 63: 63:         ...     rl_env = create_gymnasium_environment(cfg.gymnasium)
66: 65: 64: 64: 64: 64: 64: 61: 61: 46: 46: 
67: 66: 65: 65: 65: 65: 65: 62: 62: 47: 47: Performance Characteristics:
68: 67: 66: 66: 66: 66: 66: 63: 63: 48: 48:     - Factory method initialization: <10ms for typical configurations
69: 68: 67: 67: 67: 67: 67: 64: 64: 49: 49:     - Multi-agent support: Up to 100 simultaneous agents with vectorized operations
70: 69: 68: 68: 68: 68: 68: 65: 65: 50: 50:     - Real-time visualization: 30+ FPS animation performance
71: 70: 69: 69: 69: 69: 69: 66: 66: 51: 51:     - Memory efficiency: Optimized NumPy array usage for large-scale simulations
72: 71: 70: 70: 70: 70:     - RL environment step overhead: <1ms for step/reset operations
73: 72: 71: 71: 71: 71: 70: 67: 67: 52: 52: 
74: 73: 72: 72: 72: 72: 71: 68: 68: 53: 53: Backward Compatibility:
75: 74: 73: 73: 73: 73: 72: 69: 69: 54: 54:     The API maintains compatibility with legacy interfaces while providing enhanced
76: 75: 74: 74: 74: 74: 73: 70: 70: 55: 55:     Hydra-based functionality. Legacy parameter patterns are supported alongside
77: 76: 75: 75: 75: 75: 74: 71: 71: 56: 56:     new configuration-driven approaches through the from_legacy migration interface.
78: 77: 76: 76: 76: 76: 75: 72: 72: 57: 57: """
79: 78: 77: 77: 77: 77: 76: 73: 73: 58: 58: 
80: 79: 78: 78: 78: 78: 77: 74: 74: 59: 59: from typing import Union, Optional, Tuple, Any, Dict, List
81: 80: 79: 79: 79: 79: 78: 75: 75: 60: 60: import pathlib
82: 81: 80: 80: 80: 80: 79: 76: 76: 61: 61: import numpy as np
83: 82: 81: 81: 81: 81: 80: 77: 77: 62: 62: 
84: 83: 82: 82: 82: 82: 81: 78: 78: 63: 63: # Core dependency imports for type hints
85: 84: 83: 83: 83: 83: 82: 79: 79: 64: 64: try:
86: 85: 84: 84: 84: 84: 83: 80: 80: 65: 65:     from omegaconf import DictConfig
87: 86: 85: 85: 85: 85: 84: 81: 81: 66: 66:     HYDRA_AVAILABLE = True
88: 87: 86: 86: 86: 86: 85: 82: 82: 67: 67: except ImportError:
89: 88: 87: 87: 87: 87: 86: 83: 83: 68: 68:     HYDRA_AVAILABLE = False
90: 89: 88: 88: 88: 88: 87: 84: 84: 69: 69:     DictConfig = Dict[str, Any]  # Fallback type hint
91: 90: 89: 89: 89: 89: 88: 85: 85: 70: 70: 
92: 91: 90: 90: 90: 90: 89: 86: 86: 71: 71: # Import core API functions from navigation module
93: 92: 91: 91: 91: 91: 90: 87: 87: 72: 72: from .navigation import (
94: 93: 92: 92: 92: 92: 91: 88: 88: 73: 73:     create_navigator,
95: 94: 93: 93: 93: 93: 92: 89: 89: 74: 74:     create_video_plume, 
96: 95: 94: 94: 94: 94: 93: 90: 90: 75: 75:     run_plume_simulation,
97: 96: 95: 95: 95: 95: 94: 91: 91: 76: 76:     visualize_plume_simulation,
98: 97: 96: 96: 96: 96: 95: 92: 92: 77: 77:     # Legacy compatibility aliases
99: 98: 97: 97: 97: 97: 96: 93: 93: 78: 78:     create_navigator_from_config,
100: 99: 98: 98: 98: 98: 97: 94: 94: 79: 79:     create_video_plume_from_config,
101: 100: 99: 99: 99: 99: 98: 95: 95: 80: 80:     run_simulation,
102: 101: 100: 100: 100: 100: 99: 96: 96: 81: 81: )
103: 102: 101: 101: 101: 101: 100: 97: 97: 82: 82: 
104: 103: 102: 102: 102: 102: 101: 98: 98: 83: 83: # Import visualization functions from utils module
105: 104: 103: 103: 103: 103: 102: 99: 99: 84: 84: from ..utils.visualization import (
106: 105: 104: 104: 104: 104: 103: 100: 100: 85: 85:     visualize_simulation_results,
107: 106: 105: 105: 105: 105: 104: 101: 101: 86: 86:     visualize_trajectory,
108: 107: 106: 106: 106: 106: 105: 102: 102: 87: 87:     SimulationVisualization,
109: 108: 107: 107: 107: 107: 106: 103: 103: 88: 88:     batch_visualize_trajectories,
110: 109: 108: 108: 108: 108: 107: 104: 104: 89: 89:     setup_headless_mode,
111: 110: 109: 109: 109: 109: 108: 105: 105: 90: 90:     get_available_themes,
112: 111: 110: 110: 110: 110: 109: 106: 106: 91: 91:     create_simulation_visualization,
113: 112: 111: 111: 111: 111: 110: 107: 107: 92: 92:     export_animation,
114: 113: 112: 112: 112: 112: 111: 108: 108: 93: 93: )
115: 114: 113: 113: 113: 113: 112: 109: 109: 94: 94: 
116: 115: 114: 114: 114: 114: 113: 110: 110: 95: 95: # Import core protocols for type hints and advanced usage
117: 116: 115: 115: 115: 115: 114: 111: 111: 96: 96: from ..core.navigator import NavigatorProtocol
118: 117: 116: 116: 116: 116: 115: 112: 112: 97: 97: from ..environments.video_plume import VideoPlume
119: 118: 117: 117: 117: 117: 116: 113: 113: 98: 98: 
120: 119: 118: # Import Gymnasium RL integration functions with error handling
121: 120: 119: try:
122: 121: 120:     from .navigation import create_gymnasium_environment, from_legacy
123: 122: 121:     GYMNASIUM_AVAILABLE = True
124: 123: 122: except ImportError:
125: 124: 123:     # Functions not available yet - will be created by other agents
126: 125: 124:     GYMNASIUM_AVAILABLE = False
127: 126: 125:     def create_gymnasium_environment(*args, **kwargs):
128: 127: 126:         raise ImportError("create_gymnasium_environment not available - ensure Gymnasium environment is created")
129: 128: 127:     def from_legacy(*args, **kwargs):
130: 129: 128:         raise ImportError("from_legacy not available - ensure legacy migration function is created")
131: 130: 129: 118: 118: 118: 117: 114: 114: 99: 99: 
132: 131: 130: 119: 119: 119: 118: 115: 115: 100: 100: def create_navigator_instance(
133: 132: 131: 120: 120: 120: 119: 116: 116: 101: 101:     cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
134: 133: 132: 121: 121: 121: 120: 117: 117: 102: 102:     **kwargs: Any
135: 134: 133: 122: 122: 122: 121: 118: 118: 103: 103: ) -> NavigatorProtocol:
136: 135: 134: 123: 123: 123: 122: 119: 119: 104: 104:     """
137: 136: 135: 124: 124: 124: 123: 120: 120: 105: 105:     Alias for create_navigator providing enhanced documentation.
138: 137: 136: 125: 125: 125: 124: 121: 121: 106: 106:     
139: 138: 137: 126: 126: 126: 125: 122: 122: 107: 107:     This function creates navigator instances with comprehensive Hydra configuration 
140: 139: 138: 127: 127: 127: 126: 123: 123: 108: 108:     support, automatic parameter validation, and performance optimization for both 
141: 140: 139: 128: 128: 128: 127: 124: 124: 109: 109:     single-agent and multi-agent scenarios.
142: 141: 140: 129: 129: 129: 128: 125: 125: 110: 110:     
143: 142: 141: 130: 130: 130: 129: 126: 126: 111: 111:     Parameters
144: 143: 142: 131: 131: 131: 130: 127: 127: 112: 112:     ----------
145: 144: 143: 132: 132: 132: 131: 128: 128: 113: 113:     cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
146: 145: 144: 133: 133: 133: 132: 129: 129: 114: 114:         Hydra configuration object containing navigator parameters, by default None
147: 146: 145: 134: 134: 134: 133: 130: 130: 115: 115:     **kwargs : Any
148: 147: 146: 135: 135: 135: 134: 131: 131: 116: 116:         Direct parameter specification (overrides cfg values)
149: 148: 147: 136: 136: 136: 135: 132: 132: 117: 117:     
150: 149: 148: 137: 137: 137: 136: 133: 133: 118: 118:     Returns
151: 150: 149: 138: 138: 138: 137: 134: 134: 119: 119:     -------
152: 151: 150: 139: 139: 139: 138: 135: 135: 120: 120:     NavigatorProtocol
153: 152: 151: 140: 140: 140: 139: 136: 136: 121: 121:         Configured navigator instance ready for simulation use
154: 153: 152: 141: 141: 141: 140: 137: 137: 122: 122:     
155: 154: 153: 142: 142: 142: 141: 138: 138: 123: 123:     Examples
156: 155: 154: 143: 143: 143: 142: 139: 139: 124: 124:     --------
157: 156: 155: 144: 144: 144: 143: 140: 140: 125: 125:     Create single-agent navigator:
158: 157: 156: 145: 145: 145: 144: 141: 141: 126: 126:         >>> navigator = create_navigator_instance(
159: 158: 157: 146: 146: 146: 145: 142: 142: 127: 127:         ...     position=(50.0, 50.0),
160: 159: 158: 147: 147: 147: 146: 143: 143: 128: 128:         ...     orientation=45.0,
161: 160: 159: 148: 148: 148: 147: 144: 144: 129: 129:         ...     max_speed=10.0
162: 161: 160: 149: 149: 149: 148: 145: 145: 130: 130:         ... )
163: 162: 161: 150: 150: 150: 149: 146: 146: 131: 131:     
164: 163: 162: 151: 151: 151: 150: 147: 147: 132: 132:     Create with Hydra configuration:
165: 164: 163: 152: 152: 152: 151: 148: 148: 133: 133:         >>> from hydra import compose, initialize
166: 165: 164: 153: 153: 153: 152: 149: 149: 134: 134:         >>> with initialize(config_path="../conf"):
167: 166: 165: 154: 154: 154: 153: 150: 150: 135: 135:         ...     cfg = compose(config_name="config")
168: 167: 166: 155: 155: 155: 154: 151: 151: 136: 136:         ...     navigator = create_navigator_instance(cfg.navigator)
169: 168: 167: 156: 156: 156: 155: 152: 152: 137: 137:     
170: 169: 168: 157: 157: 157: 156: 153: 153: 138: 138:     Create multi-agent navigator:
171: 170: 169: 158: 158: 158: 157: 154: 154: 139: 139:         >>> navigator = create_navigator_instance(
172: 171: 170: 159: 159: 159: 158: 155: 155: 140: 140:         ...     positions=[(10, 20), (30, 40)],
173: 172: 171: 160: 160: 160: 159: 156: 156: 141: 141:         ...     orientations=[0, 90],
174: 173: 172: 161: 161: 161: 160: 157: 157: 142: 142:         ...     max_speeds=[5.0, 8.0]
175: 174: 173: 162: 162: 162: 161: 158: 158: 143: 143:         ... )
176: 175: 174: 163: 163: 163: 162: 159: 159: 144: 144:     """
177: 176: 175: 164: 164: 164: 163: 160: 160: 145: 145:     return create_navigator(cfg=cfg, **kwargs)
178: 177: 176: 165: 165: 165: 164: 161: 161: 146: 146: 
179: 178: 177: 166: 166: 166: 165: 162: 162: 147: 147: 
180: 179: 178: 167: 167: 167: 166: 163: 163: 148: 148: def create_video_plume_instance(
181: 180: 179: 168: 168: 168: 167: 164: 164: 149: 149:     cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
182: 181: 180: 169: 169: 169: 168: 165: 165: 150: 150:     **kwargs: Any
183: 182: 181: 170: 170: 170: 169: 166: 166: 151: 151: ) -> VideoPlume:
184: 183: 182: 171: 171: 171: 170: 167: 167: 152: 152:     """
185: 184: 183: 172: 172: 172: 171: 168: 168: 153: 153:     Alias for create_video_plume providing enhanced documentation.
186: 185: 184: 173: 173: 173: 172: 169: 169: 154: 154:     
187: 186: 185: 174: 174: 174: 173: 170: 170: 155: 155:     This function creates VideoPlume instances with comprehensive video processing
188: 187: 186: 175: 175: 175: 174: 171: 171: 156: 156:     capabilities, Hydra configuration integration, and automatic parameter validation.
189: 188: 187: 176: 176: 176: 175: 172: 172: 157: 157:     
190: 189: 188: 177: 177: 177: 176: 173: 173: 158: 158:     Parameters
191: 190: 189: 178: 178: 178: 177: 174: 174: 159: 159:     ----------
192: 191: 190: 179: 179: 179: 178: 175: 175: 160: 160:     cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
193: 192: 191: 180: 180: 180: 179: 176: 176: 161: 161:         Hydra configuration object containing video plume parameters, by default None
194: 193: 192: 181: 181: 181: 180: 177: 177: 162: 162:     **kwargs : Any
195: 194: 193: 182: 182: 182: 181: 178: 178: 163: 163:         Direct parameter specification (overrides cfg values)
196: 195: 194: 183: 183: 183: 182: 179: 179: 164: 164:     
197: 196: 195: 184: 184: 184: 183: 180: 180: 165: 165:     Returns
198: 197: 196: 185: 185: 185: 184: 181: 181: 166: 166:     -------
199: 198: 197: 186: 186: 186: 185: 182: 182: 167: 167:     VideoPlume
200: 199: 198: 187: 187: 187: 186: 183: 183: 168: 168:         Configured VideoPlume instance ready for simulation use
201: 200: 199: 188: 188: 188: 187: 184: 184: 169: 169:     
202: 201: 200: 189: 189: 189: 188: 185: 185: 170: 170:     Examples
203: 202: 201: 190: 190: 190: 189: 186: 186: 171: 171:     --------
204: 203: 202: 191: 191: 191: 190: 187: 187: 172: 172:     Create with direct parameters:
205: 204: 203: 192: 192: 192: 191: 188: 188: 173: 173:         >>> plume = create_video_plume_instance(
206: 205: 204: 193: 193: 193: 192: 189: 189: 174: 174:         ...     video_path="data/plume_video.mp4",
207: 206: 205: 194: 194: 194: 193: 190: 190: 175: 175:         ...     flip=True,
208: 207: 206: 195: 195: 195: 194: 191: 191: 176: 176:         ...     kernel_size=5
209: 208: 207: 196: 196: 196: 195: 192: 192: 177: 177:         ... )
210: 209: 208: 197: 197: 197: 196: 193: 193: 178: 178:     
211: 210: 209: 198: 198: 198: 197: 194: 194: 179: 179:     Create with Hydra configuration:
212: 211: 210: 199: 199: 199: 198: 195: 195: 180: 180:         >>> plume = create_video_plume_instance(cfg.video_plume)
213: 212: 211: 200: 200: 200: 199: 196: 196: 181: 181:     """
214: 213: 212: 201: 201: 201: 200: 197: 197: 182: 182:     return create_video_plume(cfg=cfg, **kwargs)
215: 214: 213: 202: 202: 202: 201: 198: 198: 183: 183: 
216: 215: 214: 203: 203: 203: 202: 199: 199: 184: 184: 
217: 216: 215: 204: 204: 204: 203: 200: 200: 185: 185: def run_navigation_simulation(
218: 217: 216: 205: 205: 205: 204: 201: 201: 186: 186:     navigator: NavigatorProtocol,
219: 218: 217: 206: 206: 206: 205: 202: 202: 187: 187:     video_plume: VideoPlume,
220: 219: 218: 207: 207: 207: 206: 203: 203: 188: 188:     cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
221: 220: 219: 208: 208: 208: 207: 204: 204: 189: 189:     **kwargs: Any
222: 221: 220: 209: 209: 209: 208: 205: 205: 190: 190: ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
223: 222: 221: 210: 210: 210: 209: 206: 206: 191: 191:     """
224: 223: 222: 211: 211: 211: 210: 207: 207: 192: 192:     Execute complete odor plume navigation simulation with comprehensive data collection.
225: 224: 223: 212: 212: 212: 211: 208: 208: 193: 193:     
226: 225: 224: 213: 213: 213: 212: 209: 209: 194: 194:     This function orchestrates frame-by-frame agent navigation through video-based 
227: 226: 225: 214: 214: 214: 213: 210: 210: 195: 195:     odor plume environments with automatic trajectory recording, performance monitoring,
228: 227: 226: 215: 215: 215: 214: 211: 211: 196: 196:     and Hydra configuration support.
229: 228: 227: 216: 216: 216: 215: 212: 212: 197: 197:     
230: 229: 228: 217: 217: 217: 216: 213: 213: 198: 198:     Parameters
231: 230: 229: 218: 218: 218: 217: 214: 214: 199: 199:     ----------
232: 231: 230: 219: 219: 219: 218: 215: 215: 200: 200:     navigator : NavigatorProtocol
233: 232: 231: 220: 220: 220: 219: 216: 216: 201: 201:         Navigator instance (SingleAgentController or MultiAgentController)
234: 233: 232: 221: 221: 221: 220: 217: 217: 202: 202:     video_plume : VideoPlume
235: 234: 233: 222: 222: 222: 221: 218: 218: 203: 203:         VideoPlume environment providing odor concentration data
236: 235: 234: 223: 223: 223: 222: 219: 219: 204: 204:     cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
237: 236: 235: 224: 224: 224: 223: 220: 220: 205: 205:         Hydra configuration object containing simulation parameters, by default None
238: 237: 236: 225: 225: 225: 224: 221: 221: 206: 206:     **kwargs : Any
239: 238: 237: 226: 226: 226: 225: 222: 222: 207: 207:         Direct parameter specification (overrides cfg values)
240: 239: 238: 227: 227: 227: 226: 223: 223: 208: 208:     
241: 240: 239: 228: 228: 228: 227: 224: 224: 209: 209:     Returns
242: 241: 240: 229: 229: 229: 228: 225: 225: 210: 210:     -------
243: 242: 241: 230: 230: 230: 229: 226: 226: 211: 211:     Tuple[np.ndarray, np.ndarray, np.ndarray]
244: 243: 242: 231: 231: 231: 230: 227: 227: 212: 212:         positions_history : Agent positions (num_agents, num_steps + 1, 2)
245: 244: 243: 232: 232: 232: 231: 228: 228: 213: 213:         orientations_history : Agent orientations (num_agents, num_steps + 1)
246: 245: 244: 233: 233: 233: 232: 229: 229: 214: 214:         odor_readings : Sensor readings (num_agents, num_steps + 1)
247: 246: 245: 234: 234: 234: 233: 230: 230: 215: 215:     
248: 247: 246: 235: 235: 235: 234: 231: 231: 216: 216:     Examples
249: 248: 247: 236: 236: 236: 235: 232: 232: 217: 217:     --------
250: 249: 248: 237: 237: 237: 236: 233: 233: 218: 218:     Basic simulation execution:
251: 250: 249: 238: 238: 238: 237: 234: 234: 219: 219:         >>> positions, orientations, readings = run_navigation_simulation(
252: 251: 250: 239: 239: 239: 238: 235: 235: 220: 220:         ...     navigator, plume, num_steps=1000, dt=0.1
253: 252: 251: 240: 240: 240: 239: 236: 236: 221: 221:         ... )
254: 253: 252: 241: 241: 241: 240: 237: 237: 222: 222:     
255: 254: 253: 242: 242: 242: 241: 238: 238: 223: 223:     Hydra-configured simulation:
256: 255: 254: 243: 243: 243: 242: 239: 239: 224: 224:         >>> results = run_navigation_simulation(
257: 256: 255: 244: 244: 244: 243: 240: 240: 225: 225:         ...     navigator, plume, cfg.simulation
258: 257: 256: 245: 245: 245: 244: 241: 241: 226: 226:         ... )
259: 258: 257: 246: 246: 246: 245: 242: 242: 227: 227:     """
260: 259: 258: 247: 247: 247: 246: 243: 243: 228: 228:     return run_plume_simulation(navigator, video_plume, cfg=cfg, **kwargs)
261: 260: 259: 248: 248: 248: 247: 244: 244: 229: 229: 
262: 261: 260: 249: 249: 249: 248: 245: 245: 230: 230: 
263: 262: 261: 250: 250: 250: 249: 246: 246: 231: 231: def visualize_results(
264: 263: 262: 251: 251: 251: 250: 247: 247: 232: 232:     positions: np.ndarray,
265: 264: 263: 252: 252: 252: 251: 248: 248: 233: 233:     orientations: np.ndarray,
266: 265: 264: 253: 253: 253: 252: 249: 249: 234: 234:     odor_readings: Optional[np.ndarray] = None,
267: 266: 265: 254: 254: 254: 253: 250: 250: 235: 235:     plume_frames: Optional[np.ndarray] = None,
268: 267: 266: 255: 255: 255: 254: 251: 251: 236: 236:     cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
269: 268: 267: 256: 256: 256: 255: 252: 252: 237: 237:     animation: bool = False,
270: 269: 268: 257: 257: 257: 256: 253: 253: 238: 238:     **kwargs: Any
271: 270: 269: 258: 258: 258: 257: 254: 254: 239: 239: ) -> "matplotlib.figure.Figure":
272: 271: 270: 259: 259: 259: 258: 255: 255: 240: 240:     """
273: 272: 271: 260: 260: 260: 259: 256: 256: 241: 241:     Create comprehensive visualizations of simulation results.
274: 273: 272: 261: 261: 261: 260: 257: 257: 242: 242:     
275: 274: 273: 262: 262: 262: 261: 258: 258: 243: 243:     This function provides unified access to both static trajectory plots and 
276: 275: 274: 263: 263: 263: 262: 259: 259: 244: 244:     animated visualizations with publication-quality formatting and Hydra 
277: 276: 275: 264: 264: 264: 263: 260: 260: 245: 245:     configuration support.
278: 277: 276: 265: 265: 265: 264: 261: 261: 246: 246:     
279: 278: 277: 266: 266: 266: 265: 262: 262: 247: 247:     Parameters
280: 279: 278: 267: 267: 267: 266: 263: 263: 248: 248:     ----------
281: 280: 279: 268: 268: 268: 267: 264: 264: 249: 249:     positions : np.ndarray
282: 281: 280: 269: 269: 269: 268: 265: 265: 250: 250:         Agent positions with shape (num_agents, num_steps, 2)
283: 282: 281: 270: 270: 270: 269: 266: 266: 251: 251:     orientations : np.ndarray
284: 283: 282: 271: 271: 271: 270: 267: 267: 252: 252:         Agent orientations with shape (num_agents, num_steps)
285: 284: 283: 272: 272: 272: 271: 268: 268: 253: 253:     odor_readings : Optional[np.ndarray], optional
286: 285: 284: 273: 273: 273: 272: 269: 269: 254: 254:         Sensor readings with shape (num_agents, num_steps), by default None
287: 286: 285: 274: 274: 274: 273: 270: 270: 255: 255:     plume_frames : Optional[np.ndarray], optional
288: 287: 286: 275: 275: 275: 274: 271: 271: 256: 256:         Video frames for background visualization, by default None
289: 288: 287: 276: 276: 276: 275: 272: 272: 257: 257:     cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
290: 289: 288: 277: 277: 277: 276: 273: 273: 258: 258:         Hydra configuration for visualization parameters, by default None
291: 290: 289: 278: 278: 278: 277: 274: 274: 259: 259:     animation : bool, optional
292: 291: 290: 279: 279: 279: 278: 275: 275: 260: 260:         Whether to create animated visualization, by default False
293: 292: 291: 280: 280: 280: 279: 276: 276: 261: 261:     **kwargs : Any
294: 293: 292: 281: 281: 281: 280: 277: 277: 262: 262:         Additional visualization parameters
295: 294: 293: 282: 282: 282: 281: 278: 278: 263: 263:     
296: 295: 294: 283: 283: 283: 282: 279: 279: 264: 264:     Returns
297: 296: 295: 284: 284: 284: 283: 280: 280: 265: 265:     -------
298: 297: 296: 285: 285: 285: 284: 281: 281: 266: 266:     matplotlib.figure.Figure
299: 298: 297: 286: 286: 286: 285: 282: 282: 267: 267:         The created matplotlib figure object
300: 299: 298: 287: 287: 287: 286: 283: 283: 268: 268:     
301: 300: 299: 288: 288: 288: 287: 284: 284: 269: 269:     Examples
302: 301: 300: 289: 289: 289: 288: 285: 285: 270: 270:     --------
303: 302: 301: 290: 290: 290: 289: 286: 286: 271: 271:     Static trajectory plot:
304: 303: 302: 291: 291: 291: 290: 287: 287: 272: 272:         >>> fig = visualize_results(positions, orientations)
305: 304: 303: 292: 292: 292: 291: 288: 288: 273: 273:     
306: 305: 304: 293: 293: 293: 292: 289: 289: 274: 274:     Animated visualization:
307: 306: 305: 294: 294: 294: 293: 290: 290: 275: 275:         >>> fig = visualize_results(
308: 307: 306: 295: 295: 295: 294: 291: 291: 276: 276:         ...     positions, orientations, 
309: 308: 307: 296: 296: 296: 295: 292: 292: 277: 277:         ...     plume_frames=frames,
310: 309: 308: 297: 297: 297: 296: 293: 293: 278: 278:         ...     animation=True
311: 310: 309: 298: 298: 298: 297: 294: 294: 279: 279:         ... )
312: 311: 310: 299: 299: 299: 298: 295: 295: 280: 280:     
313: 312: 311: 300: 300: 300: 299: 296: 296: 281: 281:     Publication-quality export:
314: 313: 312: 301: 301: 301: 300: 297: 297: 282: 282:         >>> fig = visualize_results(
315: 314: 313: 302: 302: 302: 301: 298: 298: 283: 283:         ...     positions, orientations,
316: 315: 314: 303: 303: 303: 302: 299: 299: 284: 284:         ...     cfg=viz_config,
317: 316: 315: 304: 304: 304: 303: 300: 300: 285: 285:         ...     output_path="results/trajectory.png",
318: 317: 316: 305: 305: 305: 304: 301: 301: 286: 286:         ...     show_plot=False
319: 318: 317: 306: 306: 306: 305: 302: 302: 287: 287:         ... )
320: 319: 318: 307: 307: 307: 306: 303: 303: 288: 288:     """
321: 320: 319: 308: 308: 308: 307: 304: 304: 289: 289:     if animation:
322: 321: 320: 309: 309: 309: 308: 305: 305: 290: 290:         return visualize_simulation_results(
323: 322: 321: 310: 310: 310: 309: 306: 306: 291: 291:             positions=positions,
324: 323: 322: 311: 311: 311: 310: 307: 307: 292: 292:             orientations=orientations,
325: 324: 323: 312: 312: 312: 311: 308: 308: 293: 293:             odor_readings=odor_readings,
326: 325: 324: 313: 313: 313: 312: 309: 309: 294: 294:             plume_frames=plume_frames,
327: 326: 325: 314: 314: 314: 313: 310: 310: 295: 295:             **kwargs
328: 327: 326: 315: 315: 315: 314: 311: 311: 296: 296:         )
329: 328: 327: 316: 316: 316: 315: 312: 312: 297: 297:     else:
330: 329: 328: 317: 317: 317: 316: 313: 313: 298: 298:         return visualize_trajectory(
331: 330: 329: 318: 318: 318: 317: 314: 314: 299: 299:             positions=positions,
332: 331: 330: 319: 319: 319: 318: 315: 315: 300: 300:             orientations=orientations,
333: 332: 331: 320: 320: 320: 319: 316: 316: 301: 301:             plume_frames=plume_frames,
334: 333: 332: 321: 321: 321: 320: 317: 317: 302: 302:             config=cfg,
335: 334: 333: 322: 322: 322: 321: 318: 318: 303: 303:             **kwargs
336: 335: 334: 323: 323: 323: 322: 319: 319: 304: 304:         )
337: 336: 335: 324: 324: 324: 323: 320: 320: 305: 305: 
338: 337: 336: 325: 325: 325: 324: 321: 321: 306: 306: 
339: 338: 337: 326: 326: 326: 325: 322: 322: 307: 307: # Legacy compatibility functions for backward compatibility
340: 339: 338: 327: 327: 327: 326: 323: 323: 308: 308: def create_navigator_legacy(config_path: Optional[str] = None, **kwargs: Any) -> NavigatorProtocol:
341: 340: 339: 328: 328: 328: 327: 324: 324: 309: 309:     """
342: 341: 340: 329: 329: 329: 328: 325: 325: 310: 310:     Legacy navigator creation interface for backward compatibility.
343: 342: 341: 330: 330: 330: 329: 326: 326: 311: 311:     
344: 343: 342: 331: 331: 331: 330: 327: 327: 312: 312:     This function maintains compatibility with pre-Hydra configuration patterns
345: 344: 343: 332: 332: 332: 331: 328: 328: 313: 313:     while providing access to enhanced functionality through parameter passing.
346: 345: 344: 333: 333: 333: 332: 329: 329: 314: 314:     
347: 346: 345: 334: 334: 334: 333: 330: 330: 315: 315:     Parameters
348: 347: 346: 335: 335: 335: 334: 331: 331: 316: 316:     ----------
349: 348: 347: 336: 336: 336: 335: 332: 332: 317: 317:     config_path : Optional[str], optional
350: 349: 348: 337: 337: 337: 336: 333: 333: 318: 318:         Path to configuration file (legacy parameter), by default None
351: 350: 349: 338: 338: 338: 337: 334: 334: 319: 319:     **kwargs : Any
352: 351: 350: 339: 339: 339: 338: 335: 335: 320: 320:         Navigator configuration parameters
353: 352: 351: 340: 340: 340: 339: 336: 336: 321: 321:     
354: 353: 352: 341: 341: 341: 340: 337: 337: 322: 322:     Returns
355: 354: 353: 342: 342: 342: 341: 338: 338: 323: 323:     -------
356: 355: 354: 343: 343: 343: 342: 339: 339: 324: 324:     NavigatorProtocol
357: 356: 355: 344: 344: 344: 343: 340: 340: 325: 325:         Configured navigator instance
358: 357: 356: 345: 345: 345: 344: 341: 341: 326: 326:     
359: 358: 357: 346: 346: 346: 345: 342: 342: 327: 327:     Notes
360: 359: 358: 347: 347: 347: 346: 343: 343: 328: 328:     -----
361: 360: 359: 348: 348: 348: 347: 344: 344: 329: 329:     This function is provided for backward compatibility. New code should use
362: 361: 360: 349: 349: 349: 348: 345: 345: 330: 330:     create_navigator() with Hydra configuration support.
363: 362: 361: 350: 350: 350: 349: 346: 346: 331: 331:     """
364: 363: 362: 351: 351: 351: 350: 347: 347: 332: 332:     # Convert legacy config_path to modern parameter pattern
365: 364: 363: 352: 352: 352: 351: 348: 348: 333: 333:     if config_path is not None:
366: 365: 364: 353: 353: 353: 352: 349: 349: 334: 334:         # In legacy mode, we rely on direct parameters only
367: 366: 365: 354: 354: 354: 353: 350: 350: 335: 335:         # since we can't dynamically load YAML files without Hydra context
368: 367: 366: 355: 355: 355: 354: 351: 351: 336: 336:         import warnings
369: 368: 367: 356: 356: 356: 355: 352: 352: 337: 337:         warnings.warn(
370: 369: 368: 357: 357: 357: 356: 353: 353: 338: 338:             "config_path parameter is deprecated. Use Hydra configuration or direct parameters.",
371: 370: 369: 358: 358: 358: 357: 354: 354: 339: 339:             DeprecationWarning,
372: 371: 370: 359: 359: 359: 358: 355: 355: 340: 340:             stacklevel=2
373: 372: 371: 360: 360: 360: 359: 356: 356: 341: 341:         )
374: 373: 372: 361: 361: 361: 360: 357: 357: 342: 342:     
375: 374: 373: 362: 362: 362: 361: 358: 358: 343: 343:     return create_navigator(**kwargs)
376: 375: 374: 363: 363: 363: 362: 359: 359: 344: 344: 
377: 376: 375: 364: 364: 364: 363: 360: 360: 345: 345: 
378: 377: 376: 365: 365: 365: 364: 361: 361: 346: 346: def create_video_plume_legacy(config_path: Optional[str] = None, **kwargs: Any) -> VideoPlume:
379: 378: 377: 366: 366: 366: 365: 362: 362: 347: 347:     """
380: 379: 378: 367: 367: 367: 366: 363: 363: 348: 348:     Legacy video plume creation interface for backward compatibility.
381: 380: 379: 368: 368: 368: 367: 364: 364: 349: 349:     
382: 381: 380: 369: 369: 369: 368: 365: 365: 350: 350:     Parameters
383: 382: 381: 370: 370: 370: 369: 366: 366: 351: 351:     ----------
384: 383: 382: 371: 371: 371: 370: 367: 367: 352: 352:     config_path : Optional[str], optional
385: 384: 383: 372: 372: 372: 371: 368: 368: 353: 353:         Path to configuration file (legacy parameter), by default None
386: 385: 384: 373: 373: 373: 372: 369: 369: 354: 354:     **kwargs : Any
387: 386: 385: 374: 374: 374: 373: 370: 370: 355: 355:         VideoPlume configuration parameters
388: 387: 386: 375: 375: 375: 374: 371: 371: 356: 356:     
389: 388: 387: 376: 376: 376: 375: 372: 372: 357: 357:     Returns
390: 389: 388: 377: 377: 377: 376: 373: 373: 358: 358:     -------
391: 390: 389: 378: 378: 378: 377: 374: 374: 359: 359:     VideoPlume
392: 391: 390: 379: 379: 379: 378: 375: 375: 360: 360:         Configured VideoPlume instance
393: 392: 391: 380: 380: 380: 379: 376: 376: 361: 361:     
394: 393: 392: 381: 381: 381: 380: 377: 377: 362: 362:     Notes
395: 394: 393: 382: 382: 382: 381: 378: 378: 363: 363:     -----
396: 395: 394: 383: 383: 383: 382: 379: 379: 364: 364:     This function is provided for backward compatibility. New code should use
397: 396: 395: 384: 384: 384: 383: 380: 380: 365: 365:     create_video_plume() with Hydra configuration support.
398: 397: 396: 385: 385: 385: 384: 381: 381: 366: 366:     """
399: 398: 397: 386: 386: 386: 385: 382: 382: 367: 367:     if config_path is not None:
400: 399: 398: 387: 387: 387: 386: 383: 383: 368: 368:         import warnings
401: 400: 399: 388: 388: 388: 387: 384: 384: 369: 369:         warnings.warn(
402: 401: 400: 389: 389: 389: 388: 385: 385: 370: 370:             "config_path parameter is deprecated. Use Hydra configuration or direct parameters.",
403: 402: 401: 390: 390: 390: 389: 386: 386: 371: 371:             DeprecationWarning,
404: 403: 402: 391: 391: 391: 390: 387: 387: 372: 372:             stacklevel=2
405: 404: 403: 392: 392: 392: 391: 388: 388: 373: 373:         )
406: 405: 404: 393: 393: 393: 392: 389: 389: 374: 374:     
407: 406: 405: 394: 394: 394: 393: 390: 390: 375: 375:     return create_video_plume(**kwargs)
408: 407: 406: 395: 395: 395: 394: 391: 391: 376: 376: 
409: 408: 407: 396: 396: 396: 395: 392: 392: 377: 377: 
410: 409: 408: 397: 397: 397: 396: 393: 393: 378: 378: # Module metadata and version information
411: 410: 409: 398: 398: 398: 397: 394: 394: 379: 379: __version__ = "1.0.0"
412: 411: 410: 399: 399: 399: 398: 395: 395: 380: 380: __author__ = "Odor Plume Navigation Development Team"
413: 412: 411: 400: 400: 400: 399: 396: 396: 381: 381: __description__ = "Public API for odor plume navigation library with unified package structure and Hydra configuration support"
414: 413: 412: 401: 401: 401: 400: 397: 397: 382: 382: 
415: 414: 413: 402: 402: 402: 401: 398: 398: 383: 383: # Export all public functions and classes
416: 415: 414: 403: 403: 403: 402: 399: 399: 384: 384: __all__ = [
417: 416: 415: 404: 404: 404: 403: 400: 400: 385: 385:     # Primary factory methods
418: 417: 416: 405: 405: 405: 404: 401: 401: 386: 386:     "create_navigator",
419: 418: 417: 406: 406: 406: 405: 402: 402: 387: 387:     "create_video_plume",
420: 419: 418: 407: 407: 407: 406: 403: 403: 388: 388:     "run_plume_simulation",
421: 420: 419: 408: 408: 408: 407: 404: 404: 389:     
422: 421: 420: 409: 409: 409: 408: 405: 405: 390:     # Gymnasium RL integration functions
423: 422: 421: 410: 410: 410: 409: 406: 406: 391:     "create_gymnasium_environment",
424: 423: 422: 411: 411: 411: 410: 407: 407: 392:     "from_legacy",
425: 424: 423: 412: 412: 412: 411: 408: 408: 393: 389:     
426: 425: 424: 413: 413: 413: 412: 409: 409: 394: 390:     # Enhanced API aliases
427: 426: 425: 414: 414: 414: 413: 410: 410: 395: 391:     "create_navigator_instance", 
428: 427: 426: 415: 415: 415: 414: 411: 411: 396: 392:     "create_video_plume_instance",
429: 428: 427: 416: 416: 416: 415: 412: 412: 397: 393:     "run_navigation_simulation",
430: 429: 428: 417: 417: 417: 416: 413: 413: 398: 394:     "visualize_results",
431: 430: 429: 418: 418: 418: 417: 414: 414: 399: 395:     
432: 431: 430: 419: 419: 419: 418: 415: 415: 400: 396:     # Visualization functions
433: 432: 431: 420: 420: 420: 419: 416: 416: 401: 397:     "visualize_simulation_results",
434: 433: 432: 421: 421: 421: 420: 417: 417: 402: 398:     "visualize_trajectory", 
435: 434: 433: 422: 422: 422: 421: 418: 418: 403: 399:     "visualize_plume_simulation",
436: 435: 434: 423: 423: 423: 422: 419: 419: 404: 400:     "SimulationVisualization",
437: 436: 435: 424: 424: 424: 423: 420: 420: 405: 401:     "batch_visualize_trajectories",
438: 437: 436: 425: 425: 425: 424: 421: 421: 406: 402:     "setup_headless_mode",
439: 438: 437: 426: 426: 426: 425: 422: 422: 407: 403:     "get_available_themes",
440: 439: 438: 427: 427: 427: 426: 423: 423: 408: 404:     "create_simulation_visualization",
441: 440: 439: 428: 428: 428: 427: 424: 424: 409: 405:     "export_animation",
442: 441: 440: 429: 429: 429: 428: 425: 425: 410: 406:     
443: 442: 441: 430: 430: 430: 429: 426: 426: 411: 407:     # Core protocols and classes for advanced usage
444: 443: 442: 431: 431: 431: 430: 427: 427: 412: 408:     "NavigatorProtocol",
445: 444: 443: 432: 432: 432: 431: 428: 428: 413: 409:     "VideoPlume",
446: 445: 444: 433: 433: 433: 432: 429: 429: 414: 410:     
447: 446: 445: 434: 434: 434: 433: 430: 430: 415: 411:     # Legacy compatibility functions
448: 447: 446: 435: 435: 435: 434: 431: 431: 416: 412:     "create_navigator_from_config",
449: 448: 447: 436: 436: 436: 435: 432: 432: 417: 413:     "create_video_plume_from_config", 
450: 449: 448: 437: 437: 437: 436: 433: 433: 418: 414:     "run_simulation",
451: 450: 449: 438: 438: 438: 437: 434: 434: 419: 415:     "create_navigator_legacy",
452: 451: 450: 439: 439: 439: 438: 435: 435: 420: 416:     "create_video_plume_legacy",
453: 452: 451: 440: 440: 440: 439: 436: 436: 421: 417:     
454: 453: 452: 441: 441: 441: 440: 437: 437: 422: 418:     # Module metadata
455: 454: 453: 442: 442: 442: 441: 438: 438: 423: 419:     "__version__",
456: 455: 454: 443: 443: 443: 442: 439: 439: 424: 420:     "__author__",
457: 456: 455: 444: 444: 444: 443: 440: 440: 425: 421:     "__description__",
458: 457: 456: 445: 445: 445: 444: 441: 441: 426: 422: ]
459: 458: 457: 446: 446: 446: 445: 442: 442: 427: 423: 
460: 459: 458: 447: 447: 447: 446: 443: 443: 428: 424: # Conditional exports based on Hydra availability
461: 460: 459: 448: 448: 448: 447: 444: 444: 429: 425: if HYDRA_AVAILABLE:
462: 461: 460: 449: 449: 449: 448: 445: 445: 430: 426:     # Add Hydra-specific functionality to exports
463: 462: 461: 450: 450: 450: 449: 446: 446: 431: 427:     __all__.extend([
464: 463: 462: 451: 451: 451: 450: 447: 447: 432: 428:         "DictConfig",  # Re-export for type hints
465: 464: 463: 452: 452: 452: 451: 448: 448: 433: 429:     ])
466: 465: 464: 453: 453: 453: 452: 449: 449: 434: 430: 
467: 466: 465: 454: 454: 454: 453: 450: 450: 435: 431: # Package initialization message (optional, for development/debugging)
468: 467: 466: 455: 455: 455: 454: 451: 451: 436: 432: def _get_api_info() -> Dict[str, Any]:
469: 468: 467: 456: 456: 456: 455: 452: 452: 437: 433:     """Get API module information for debugging and introspection."""
470: 469: 468: 457: 457: 457: 456: 453: 453: 438: 434:     return {
471: 470: 469: 458: 458: 458: 457: 454: 454: 439: 435:         "version": __version__,
472: 471: 470: 459: 459: 459: 458: 455: 455: 440: 436:         "hydra_available": HYDRA_AVAILABLE,
473: 472: 471: 460: 460: 460: 459: 456: 456: 441: 437:         "public_functions": len(__all__),
474: 473: 472: 461: 461: 461: 460: 457: 457: 442: 438:         "primary_functions": [
475: 474: 473: 462: 462: 462: 461: 458: 458: 443: 439:             "create_navigator", 
476: 475: 474: 463: 463: 463: 462: 459: 459: 444: 440:             "create_video_plume", 
477: 476: 475: 464: 464: 464: 463: 460: 460: 445: 441:             "run_plume_simulation",
478: 477: 476: 465: 465: 465: 464: 461: 461: 446: 442:             "visualize_simulation_results"
479: 478: 477: 466: 466: 466: 465: 462: 462: 447: 443:         ],
480: 479: 478: 467: 467: 467: 466: 463: 463: 448: 444:         "legacy_support": True,
481: 480: 479: 468: 468: 468: 467: 464: 464: 449: 445:         "configuration_types": ["direct_parameters", "hydra_dictconfig"] + (
482: 481: 480: 469: 469: 469: 468: 465: 465: 450: 446:             ["yaml_files"] if HYDRA_AVAILABLE else []
483: 482: 481: 470: 470: 470: 469: 466: 466: 451: 447:         )
484: 483: 482: 471: 471: 471: 470: 467: 467: 452: 448:     }
485: 484: 483: 472: 472: 472: 471: 468: 468: 453: 449: 
486: 485: 484: 473: 473: 473: 472: 469: 469: 454: 450: 
487: 486: 485: 474: 474: 474: 473: 470: 470: 455: 451: # Optional: Expose API info for debugging
488: 487: 486: 475: 475: 475: 474: 471: 471: 456: 452: def get_api_info() -> Dict[str, Any]:
489: 488: 487: 476: 476: 476: 475: 472: 472: 457: 453:     """
490: 489: 488: 477: 477: 477: 476: 473: 473: 458: 454:     Get comprehensive information about the API module.
491: 490: 489: 478: 478: 478: 477: 474: 474: 459: 455:     
492: 491: 490: 479: 479: 479: 478: 475: 475: 460: 456:     This function provides metadata about available functions, configuration
493: 492: 491: 480: 480: 480: 479: 476: 476: 461: 457:     options, and system capabilities for debugging and introspection.
494: 493: 492: 481: 481: 481: 480: 477: 477: 462: 458:     
495: 494: 493: 482: 482: 482: 481: 478: 478: 463: 459:     Returns
496: 495: 494: 483: 483: 483: 482: 479: 479: 464: 460:     -------
497: 496: 495: 484: 484: 484: 483: 480: 480: 465: 461:     Dict[str, Any]
498: 497: 496: 485: 485: 485: 484: 481: 481: 466: 462:         Dictionary containing API module information
499: 498: 497: 486: 486: 486: 485: 482: 482: 467: 463:     
500: 499: 498: 487: 487: 487: 486: 483: 483: 468: 464:     Examples
501: 500: 499: 488: 488: 488: 487: 484: 484: 469: 465:     --------
502: 501: 500: 489: 489: 489: 488: 485: 485: 470: 466:     >>> from odor_plume_nav.api import get_api_info
503: 502: 501: 490: 490: 490: 489: 486: 486: 471: 467:     >>> info = get_api_info()
504: 503: 502: 491: 491: 491: 490: 487: 487: 472: 468:     >>> print(f"API version: {info['version']}")
505: 504: 503: 492: 492: 492: 491: 488: 488: 473: 469:     >>> print(f"Hydra support: {info['hydra_available']}")
506: 505: 504: 493: 493: 493: 492: 489: 489: 474: 470:     """
507: 506: 505: 494: 494: 494: 493: 490: 490: 475: 471:     return _get_api_info()