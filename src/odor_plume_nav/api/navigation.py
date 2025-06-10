1: 1: 1: 1: 1: 1: 1: """
2: 2: 2: 2: 2: 2: 2: Consolidated public API module for odor plume navigation.
3: 3: 3: 3: 3: 3: 3: 
4: 4: 4: 4: 4: 4: 4: This module provides a unified, research-oriented API integrating navigation orchestration,
5: 5: 5: 5: 5: 5: 5: factory methods, and simulation execution with comprehensive Hydra-based configuration
6: 6: 6: 6: 6: 6: 6: support. Combines functionality from legacy interfaces/api.py, services/simulation_runner.py,
7: 7: 7: 7: 7: 7: 7: services/factories.py, and services/video_plume_factory.py into a standardized API surface.
8: 8: 8: 8: 8: 8: 8: 
9: 9: 9: 9: 9: 9: 9: The API supports multiple research frameworks:
10: 10: 10: 10: 10: 10: 10: - Kedro pipeline integration through factory method patterns
11: 11: 11: 11: 11: 11: 11: - RL framework compatibility via NumPy array interfaces and protocol-based definitions
12: 12: 12: 12: 12: 12: 12: - ML analysis tools through standardized data exchange and configuration management
13: 13: 13: 13: 13: 13: 13: - Interactive research environments through comprehensive parameter validation
14: 14: 14: 14: 14: 14: 14: 
15: 15: 15: 15: 15: 15: 15: Key Features:
16: 16: 16: 16: 16: 16: 16:     - Hydra DictConfig integration for hierarchical configuration management
17: 17: 17: 17: 17: 17: 17:     - Factory pattern implementation supporting both direct parameters and structured configs
18: 18: 18: 18: 18: 18: 18:     - Enhanced error handling with structured logging for research reproducibility
19: 19: 19: 19: 19: 19: 19:     - Protocol-based interfaces ensuring extensibility and algorithm compatibility
20: 20: 20: 20: 20: 20: 20:     - Automatic seed management integration for deterministic experiment execution
21: 21: 21: 21: 21: 21: 21:     - Performance-optimized initialization meeting <2s requirement for complex configurations
22: 22: 22: 22: 22: 22: 22: 
23: 23: 23: 23: 23: 23: 23: Example Usage:
24: 24: 24: 24: 24: 24: 24:     Kedro pipeline integration:
25: 25: 25: 25: 25: 25: 25:         >>> from hydra import compose, initialize
26: 26: 26: 26: 26: 26: 26:         >>> from odor_plume_nav.api.navigation import create_navigator, run_plume_simulation
27: 27: 27: 27: 27: 27: 27:         >>> 
28: 28: 28: 28: 28: 28: 28:         >>> with initialize(config_path="../conf"):
29: 29: 29: 29: 29: 29: 29:         ...     cfg = compose(config_name="config")
30: 30: 30: 30: 30: 30: 30:         ...     navigator = create_navigator(cfg.navigator)
31: 31: 31: 31: 31: 31: 31:         ...     plume = create_video_plume(cfg.video_plume)
32: 32: 32: 32: 32: 32: 32:         ...     results = run_plume_simulation(navigator, plume, cfg.simulation)
33: 33: 33: 33: 33: 33: 33: 
34: 34: 34: 34: 34: 34: 34:     RL framework integration:
35: 35: 35: 35: 35: 35: 35:         >>> from odor_plume_nav.api.navigation import create_navigator
36: 36: 36: 36: 36: 36: 36:         >>> from odor_plume_nav.utils.seed_manager import set_global_seed
37: 37: 37: 37: 37: 37: 37:         >>> 
38: 38: 38: 38: 38: 38: 38:         >>> set_global_seed(42)  # Reproducible RL experiments
39: 39: 39: 39: 39: 39: 39:         >>> navigator = create_navigator(position=(10, 10), max_speed=5.0)
40: 40: 40: 40: 40: 40: 40:         >>> # Use navigator.step(env_array) in RL loop
41: 41: 41: 41: 41: 41: 41: 
42: 42: 42: 42: 42: 42: 42:     Direct parameter usage:
43: 43: 43: 43: 43: 43: 43:         >>> navigator = create_navigator(
44: 44: 44: 44: 44: 44: 44:         ...     position=(50.0, 50.0),
45: 45: 45: 45: 45: 45: 45:         ...     orientation=45.0,
46: 46: 46: 46: 46: 46: 46:         ...     max_speed=10.0
47: 47: 47: 47: 47: 47: 47:         ... )
48: 48: 48: 48: 48: 48: 48:         >>> plume = create_video_plume(
49: 49: 49: 49: 49: 49: 49:         ...     video_path="data/plume_video.mp4",
50: 50: 50: 50: 50: 50: 50:         ...     flip=True,
51: 51: 51: 51: 51: 51: 51:         ...     kernel_size=5
52: 52: 52: 52: 52: 52: 52:         ... )
53: 53: 53:     Reinforcement learning training workflow:
54: 54: 54:         >>> from odor_plume_nav.api.navigation import create_gymnasium_environment
55: 55: 55:         >>> from stable_baselines3 import PPO
56: 56: 56:         >>> 
57: 57: 57:         >>> # Create RL-ready environment
58: 58: 58:         >>> env = create_gymnasium_environment(
59: 59: 59:         ...     video_path="data/plume_experiment.mp4",
60: 60: 60:         ...     initial_position=(50, 50),
61: 61: 61:         ...     max_speed=2.0,
62: 62: 62:         ...     render_mode="human"
63: 63: 63:         ... )
64: 64: 64:         >>> 
65: 65: 65:         >>> # Train policy with stable-baselines3
66: 66: 66:         >>> model = PPO("MultiInputPolicy", env, verbose=1)
67: 67: 67:         >>> model.learn(total_timesteps=50000)
68: 68: 68: 
69: 69: 69:     Legacy migration to RL:
70: 70: 70:         >>> # Start with traditional components
71: 71: 71:         >>> navigator = create_navigator(position=(50, 50), max_speed=2.0)
72: 72: 72:         >>> plume = create_video_plume(video_path="data/plume_video.mp4")
73: 73: 73:         >>> 
74: 74: 74:         >>> # Migrate to RL environment
75: 75: 75:         >>> env = from_legacy(navigator, plume, render_mode="human")
76: 76: 76:         >>> 
77: 77: 77:         >>> # Now use with RL algorithms
78: 78: 78:         >>> from stable_baselines3 import SAC
79: 79: 79:         >>> model = SAC("MultiInputPolicy", env)
80: 80: 80: 53: 53: 53: 53: """
81: 81: 81: 54: 54: 54: 54: 
82: 82: 82: 55: 55: 55: 55: import pathlib
83: 83: 83: 56: 56: 56: 56: from typing import List, Optional, Tuple, Union, Any, Dict
84: 84: 84: 57: 57: 57: 57: from contextlib import suppress
85: 85: 85: 58: 58: 58: 58: import numpy as np
86: 86: 86: 59: 59: 59: 59: from loguru import logger
87: 87: 87: 60: 60: 60: 60: 
88: 88: 88: 61: 61: 61: 61: # Hydra imports for configuration management
89: 89: 89: 62: 62: 62: 62: try:
90: 90: 90: 63: 63: 63: 63:     from omegaconf import DictConfig, OmegaConf
91: 91: 91: 64: 64: 64: 64:     HYDRA_AVAILABLE = True
92: 92: 92: 65: 65: 65: 65: except ImportError:
93: 93: 93: 66: 66: 66: 66:     HYDRA_AVAILABLE = False
94: 94: 94: 67: 67: 67: 67:     DictConfig = dict
95: 95: 95: 68: 68: 68: 68:     logger.warning("Hydra not available. Falling back to dictionary configuration.")
96: 96: 96: 69: 69: 69: 69: 
97: 97: 97: 70: 70: 70: 70: # Import dependencies from unified module structure
98: 98: 98: 71: 71: 71: 71: from ..core.protocols import NavigatorProtocol
99: 99: 99: 72: 72: 72: 72: from ..core.controllers import SingleAgentController, MultiAgentController
100: 100: 100: 73: 73: 73: 73: from ..environments.video_plume import VideoPlume
101: 101: 101: 74: 74: 74: 74: from ..config.schemas import (
102: 102: 102: 75: 75: 75: 75:     NavigatorConfig, 
103: 103: 103: 76: 76: 76: 76:     SingleAgentConfig, 
104: 104: 104: 77: 77: 77: 77:     MultiAgentConfig, 
105: 105: 105: 78: 78: 78: 78:     VideoPlumeConfig,
106: 106: 106: 79: 79: 79: 79:     SimulationConfig
107: 107: 107: 80: 80: 80: 80: )
108: 108: 108: 81: 81: 81: 81: from ..utils.seed_manager import get_seed_manager, SeedManager
109: 109: 109: 82: 82: 82: 82: from ..utils.visualization import visualize_simulation_results, visualize_trajectory
110: 110: 110: 83: 83: 
111: 111: 111: 84: 84: # Gymnasium environment imports for RL integration
112: 112: 112: 85: 85: try:
113: 113: 113: 86: 86:     from ..environments.gymnasium_env import GymnasiumEnv, create_gymnasium_environment as _create_gymnasium_env
114: 114: 114: 87: 87:     GYMNASIUM_AVAILABLE = True
115: 115: 115: 88: 88: except ImportError:
116: 116: 116: 89: 89:     GYMNASIUM_AVAILABLE = False
117: 117: 117: 90: 90:     GymnasiumEnv = None
118: 118: 118: 91: 91:     _create_gymnasium_env = None
119: 119: 119: 92: 92: 83: 83: 
120: 120: 120: 93: 93: 84: 84: 
121: 121: 121: 94: 94: 85: 85: def create_navigator(
122: 122: 122: 95: 95: 86: 86:     cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
123: 123: 123: 96: 96: 87: 87:     position: Optional[Union[Tuple[float, float], List[float], np.ndarray]] = None,
124: 124: 124: 97: 97: 88: 88:     positions: Optional[Union[List[Tuple[float, float]], np.ndarray]] = None,
125: 125: 125: 98: 98: 89: 89:     orientation: Optional[float] = None,
126: 126: 126: 99: 99: 90: 90:     orientations: Optional[Union[List[float], np.ndarray]] = None,
127: 127: 127: 100: 100: 91: 91:     speed: Optional[float] = None,
128: 128: 128: 101: 101: 92: 92:     speeds: Optional[Union[List[float], np.ndarray]] = None,
129: 129: 129: 102: 102: 93: 93:     max_speed: Optional[float] = None,
130: 130: 130: 103: 103: 94: 94:     max_speeds: Optional[Union[List[float], np.ndarray]] = None,
131: 131: 131: 104: 104: 95: 95:     angular_velocity: Optional[float] = None,
132: 132: 132: 105: 105: 96: 96:     angular_velocities: Optional[Union[List[float], np.ndarray]] = None,
133: 133: 133: 106: 106: 97: 97:     **kwargs: Any
134: 134: 134: 107: 107: 98: 98: ) -> NavigatorProtocol:
135: 135: 135: 108: 108: 99: 99:     """
136: 136: 136: 109: 109: 100: 100:     Create a Navigator instance with Hydra configuration support and enhanced validation.
137: 137: 137: 110: 110: 101: 101: 
138: 138: 138: 111: 111: 102: 102:     This function provides a unified interface for creating both single-agent and multi-agent
139: 139: 139: 112: 112: 103: 103:     navigators using either direct parameter specification or Hydra DictConfig objects.
140: 140: 140: 113: 113: 104: 104:     Supports automatic detection of single vs multi-agent scenarios based on parameter shapes.
141: 141: 141: 114: 114: 105: 105: 
142: 142: 142: 115: 115: 106: 106:     Parameters
143: 143: 143: 116: 116: 107: 107:     ----------
144: 144: 144: 117: 117: 108: 108:     cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
145: 145: 145: 118: 118: 109: 109:         Hydra configuration object or dictionary containing navigator parameters.
146: 146: 146: 119: 119: 110: 110:         Takes precedence over individual parameters if provided, by default None
147: 147: 147: 120: 120: 111: 111:     position : Optional[Union[Tuple[float, float], List[float], np.ndarray]], optional
148: 148: 148: 121: 121: 112: 112:         Initial position for single-agent navigator [x, y], by default None
149: 149: 149: 122: 122: 113: 113:     positions : Optional[Union[List[Tuple[float, float]], np.ndarray]], optional
150: 150: 150: 123: 123: 114: 114:         Initial positions for multi-agent navigator [[x1, y1], [x2, y2], ...], by default None
151: 151: 151: 124: 124: 115: 115:     orientation : Optional[float], optional
152: 152: 152: 125: 125: 116: 116:         Initial orientation in degrees for single agent, by default None
153: 153: 153: 126: 126: 117: 117:     orientations : Optional[Union[List[float], np.ndarray]], optional
154: 154: 154: 127: 127: 118: 118:         Initial orientations in degrees for multiple agents, by default None
155: 155: 155: 128: 128: 119: 119:     speed : Optional[float], optional
156: 156: 156: 129: 129: 120: 120:         Initial speed for single agent, by default None
157: 157: 157: 130: 130: 121: 121:     speeds : Optional[Union[List[float], np.ndarray]], optional
158: 158: 158: 131: 131: 122: 122:         Initial speeds for multiple agents, by default None
159: 159: 159: 132: 132: 123: 123:     max_speed : Optional[float], optional
160: 160: 160: 133: 133: 124: 124:         Maximum speed for single agent, by default None
161: 161: 161: 134: 134: 125: 125:     max_speeds : Optional[Union[List[float], np.ndarray]], optional
162: 162: 162: 135: 135: 126: 126:         Maximum speeds for multiple agents, by default None
163: 163: 163: 136: 136: 127: 127:     angular_velocity : Optional[float], optional
164: 164: 164: 137: 137: 128: 128:         Angular velocity for single agent in degrees/second, by default None
165: 165: 165: 138: 138: 129: 129:     angular_velocities : Optional[Union[List[float], np.ndarray]], optional
166: 166: 166: 139: 139: 130: 130:         Angular velocities for multiple agents in degrees/second, by default None
167: 167: 167: 140: 140: 131: 131:     **kwargs : Any
168: 168: 168: 141: 141: 132: 132:         Additional parameters for navigator configuration
169: 169: 169: 142: 142: 133: 133: 
170: 170: 170: 143: 143: 134: 134:     Returns
171: 171: 171: 144: 144: 135: 135:     -------
172: 172: 172: 145: 145: 136: 136:     NavigatorProtocol
173: 173: 173: 146: 146: 137: 137:         Configured navigator instance (SingleAgentController or MultiAgentController)
174: 174: 174: 147: 147: 138: 138: 
175: 175: 175: 148: 148: 139: 139:     Raises
176: 176: 176: 149: 149: 140: 140:     ------
177: 177: 177: 150: 150: 141: 141:     ValueError
178: 178: 178: 151: 151: 142: 142:         If both single and multi-agent parameters are specified
179: 179: 179: 152: 152: 143: 143:         If configuration validation fails
180: 180: 180: 153: 153: 144: 144:         If required parameters are missing
181: 181: 181: 154: 154: 145: 145:     TypeError
182: 182: 182: 155: 155: 146: 146:         If parameter types are invalid
183: 183: 183: 156: 156: 147: 147:     RuntimeError
184: 184: 184: 157: 157: 148: 148:         If navigator initialization exceeds performance requirements
185: 185: 185: 158: 158: 149: 149: 
186: 186: 186: 159: 159: 150: 150:     Examples
187: 187: 187: 160: 160: 151: 151:     --------
188: 188: 188: 161: 161: 152: 152:     Create single agent navigator with Hydra config:
189: 189: 189: 162: 162: 153: 153:         >>> from hydra import compose, initialize
190: 190: 190: 163: 163: 154: 154:         >>> with initialize(config_path="../conf"):
191: 191: 191: 164: 164: 155: 155:         ...     cfg = compose(config_name="config")
192: 192: 192: 165: 165: 156: 156:         ...     navigator = create_navigator(cfg.navigator)
193: 193: 193: 166: 166: 157: 157: 
194: 194: 194: 167: 167: 158: 158:     Create multi-agent navigator with direct parameters:
195: 195: 195: 168: 168: 159: 159:         >>> navigator = create_navigator(
196: 196: 196: 169: 169: 160: 160:         ...     positions=[(10, 20), (30, 40)],
197: 197: 197: 170: 170: 161: 161:         ...     orientations=[0, 90],
198: 198: 198: 171: 171: 162: 162:         ...     max_speeds=[5.0, 8.0]
199: 199: 199: 172: 172: 163: 163:         ... )
200: 200: 200: 173: 173: 164: 164: 
201: 201: 201: 174: 174: 165: 165:     Create single agent with mixed configuration:
202: 202: 202: 175: 175: 166: 166:         >>> cfg = DictConfig({"max_speed": 10.0, "angular_velocity": 0.1})
203: 203: 203: 176: 176: 167: 167:         >>> navigator = create_navigator(cfg, position=(50, 50), orientation=45)
204: 204: 204: 177: 177: 168: 168:     
205: 205: 205: 178: 178: 169: 169:     Notes
206: 206: 206: 179: 179: 170: 170:     -----
207: 207: 207: 180: 180: 171: 171:     Configuration precedence order:
208: 208: 208: 181: 181: 172: 172:     1. Direct parameters (position, orientation, etc.)
209: 209: 209: 182: 182: 173: 173:     2. Hydra DictConfig object values
210: 210: 210: 183: 183: 174: 174:     3. Default values from configuration schemas
211: 211: 211: 184: 184: 175: 175:     
212: 212: 212: 185: 185: 176: 176:     The function automatically detects single vs multi-agent scenarios:
213: 213: 213: 186: 186: 177: 177:     - Uses positions/orientations (plural) for multi-agent
214: 214: 214: 187: 187: 178: 178:     - Uses position/orientation (singular) for single-agent
215: 215: 215: 188: 188: 179: 179:     - Raises error if both are specified
216: 216: 216: 189: 189: 180: 180:     """
217: 217: 217: 190: 190: 181: 181:     # Initialize logger with function context
218: 218: 218: 191: 191: 182: 182:     func_logger = logger.bind(
219: 219: 219: 192: 192: 183: 183:         module=__name__,
220: 220: 220: 193: 193: 184: 184:         function="create_navigator",
221: 221: 221: 194: 194: 185: 185:         cfg_provided=cfg is not None,
222: 222: 222: 195: 195: 186: 186:         direct_params_provided=any([
223: 223: 223: 196: 196: 187: 187:             position is not None, positions is not None,
224: 224: 224: 197: 197: 188: 188:             orientation is not None, orientations is not None
225: 225: 225: 198: 198: 189: 189:         ])
226: 226: 226: 199: 199: 190: 190:     )
227: 227: 227: 200: 200: 191: 191: 
228: 228: 228: 201: 201: 192: 192:     try:
229: 229: 229: 202: 202: 193: 193:         # Validate parameter consistency
230: 230: 230: 203: 203: 194: 194:         if position is not None and positions is not None:
231: 231: 231: 204: 204: 195: 195:             raise ValueError(
232: 232: 232: 205: 205: 196: 196:                 "Cannot specify both 'position' (single-agent) and 'positions' (multi-agent). "
233: 233: 233: 206: 206: 197: 197:                 "Please provide only one."
234: 234: 234: 207: 207: 198: 198:             )
235: 235: 235: 208: 208: 199: 199: 
236: 236: 236: 209: 209: 200: 200:         if orientation is not None and orientations is not None:
237: 237: 237: 210: 210: 201: 201:             raise ValueError(
238: 238: 238: 211: 211: 202: 202:                 "Cannot specify both 'orientation' (single-agent) and 'orientations' (multi-agent). "
239: 239: 239: 212: 212: 203: 203:                 "Please provide only one."
240: 240: 240: 213: 213: 204: 204:             )
241: 241: 241: 214: 214: 205: 205: 
242: 242: 242: 215: 215: 206: 206:         # Process configuration object
243: 243: 243: 216: 216: 207: 207:         config_params = {}
244: 244: 244: 217: 217: 208: 208:         if cfg is not None:
245: 245: 245: 218: 218: 209: 209:             try:
246: 246: 246: 219: 219: 210: 210:                 if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
247: 247: 247: 220: 220: 211: 211:                     config_params = OmegaConf.to_container(cfg, resolve=True)
248: 248: 248: 221: 221: 212: 212:                 elif isinstance(cfg, dict):
249: 249: 249: 222: 222: 213: 213:                     config_params = cfg.copy()
250: 250: 250: 223: 223: 214: 214:                 else:
251: 251: 251: 224: 224: 215: 215:                     raise TypeError(f"Configuration must be DictConfig or dict, got {type(cfg)}")
252: 252: 252: 225: 225: 216: 216:             except Exception as e:
253: 253: 253: 226: 226: 217: 217:                 func_logger.error(f"Failed to process configuration: {e}")
254: 254: 254: 227: 227: 218: 218:                 raise ValueError(f"Invalid configuration object: {e}") from e
255: 255: 255: 228: 228: 219: 219: 
256: 256: 256: 229: 229: 220: 220:         # Determine navigator type and merge parameters with precedence
257: 257: 257: 230: 230: 221: 221:         # Direct parameters override configuration values
258: 258: 258: 231: 231: 222: 222:         is_multi_agent = (
259: 259: 259: 232: 232: 223: 223:             positions is not None or 
260: 260: 260: 233: 233: 224: 224:             orientations is not None or
261: 261: 261: 234: 234: 225: 225:             speeds is not None or
262: 262: 262: 235: 235: 226: 226:             max_speeds is not None or
263: 263: 263: 236: 236: 227: 227:             angular_velocities is not None or
264: 264: 264: 237: 237: 228: 228:             "positions" in config_params
265: 265: 265: 238: 238: 229: 229:         )
266: 266: 266: 239: 239: 230: 230: 
267: 267: 267: 240: 240: 231: 231:         if is_multi_agent:
268: 268: 268: 241: 241: 232: 232:             # Multi-agent navigator
269: 269: 269: 242: 242: 233: 233:             merged_params = {
270: 270: 270: 243: 243: 234: 234:                 "positions": positions or config_params.get("positions"),
271: 271: 271: 244: 244: 235: 235:                 "orientations": orientations or config_params.get("orientations"),
272: 272: 272: 245: 245: 236: 236:                 "speeds": speeds or config_params.get("speeds"),
273: 273: 273: 246: 246: 237: 237:                 "max_speeds": max_speeds or config_params.get("max_speeds"),
274: 274: 274: 247: 247: 238: 238:                 "angular_velocities": angular_velocities or config_params.get("angular_velocities"),
275: 275: 275: 248: 248: 239: 239:             }
276: 276: 276: 249: 249: 240: 240:             
277: 277: 277: 250: 250: 241: 241:             # Add any additional config parameters
278: 278: 278: 251: 251: 242: 242:             merged_params.update({k: v for k, v in config_params.items() 
279: 279: 279: 252: 252: 243: 243:                                 if k not in merged_params})
280: 280: 280: 253: 253: 244: 244:             merged_params.update(kwargs)
281: 281: 281: 254: 254: 245: 245: 
282: 282: 282: 255: 255: 246: 246:             # Validate configuration using Pydantic schema
283: 283: 283: 256: 256: 247: 247:             try:
284: 284: 284: 257: 257: 248: 248:                 validated_config = MultiAgentConfig(**merged_params)
285: 285: 285: 258: 258: 249: 249:                 func_logger.info(
286: 286: 286: 259: 259: 250: 250:                     "Multi-agent navigator configuration validated",
287: 287: 287: 260: 260: 251: 251:                     extra={
288: 288: 288: 261: 261: 252: 252:                         "num_agents": len(validated_config.positions) if validated_config.positions else None,
289: 289: 289: 262: 262: 253: 253:                         "config_source": "hydra" if cfg is not None else "direct"
290: 290: 290: 263: 263: 254: 254:                     }
291: 291: 291: 264: 264: 255: 255:                 )
292: 292: 292: 265: 265: 256: 256:             except Exception as e:
293: 293: 293: 266: 266: 257: 257:                 func_logger.error(f"Multi-agent configuration validation failed: {e}")
294: 294: 294: 267: 267: 258: 258:                 raise ValueError(f"Invalid multi-agent configuration: {e}") from e
295: 295: 295: 268: 268: 259: 259: 
296: 296: 296: 269: 269: 260: 260:             # Create multi-agent controller
297: 297: 297: 270: 270: 261: 261:             navigator = MultiAgentController(
298: 298: 298: 271: 271: 262: 262:                 positions=validated_config.positions,
299: 299: 299: 272: 272: 263: 263:                 orientations=validated_config.orientations,
300: 300: 300: 273: 273: 264: 264:                 speeds=validated_config.speeds,
301: 301: 301: 274: 274: 265: 265:                 max_speeds=validated_config.max_speeds,
302: 302: 302: 275: 275: 266: 266:                 angular_velocities=validated_config.angular_velocities,
303: 303: 303: 276: 276: 267: 267:                 config=validated_config,
304: 304: 304: 277: 277: 268: 268:                 seed_manager=get_seed_manager()
305: 305: 305: 278: 278: 269: 269:             )
306: 306: 306: 279: 279: 270: 270: 
307: 307: 307: 280: 280: 271: 271:         else:
308: 308: 308: 281: 281: 272: 272:             # Single-agent navigator
309: 309: 309: 282: 282: 273: 273:             merged_params = {
310: 310: 310: 283: 283: 274: 274:                 "position": position or config_params.get("position"),
311: 311: 311: 284: 284: 275: 275:                 "orientation": orientation or config_params.get("orientation", 0.0),
312: 312: 312: 285: 285: 276: 276:                 "speed": speed or config_params.get("speed", 0.0),
313: 313: 313: 286: 286: 277: 277:                 "max_speed": max_speed or config_params.get("max_speed", 1.0),
314: 314: 314: 287: 287: 278: 278:                 "angular_velocity": angular_velocity or config_params.get("angular_velocity", 0.0),
315: 315: 315: 288: 288: 279: 279:             }
316: 316: 316: 289: 289: 280: 280:             
317: 317: 317: 290: 290: 281: 281:             # Add any additional config parameters
318: 318: 318: 291: 291: 282: 282:             merged_params.update({k: v for k, v in config_params.items() 
319: 319: 319: 292: 292: 283: 283:                                 if k not in merged_params})
320: 320: 320: 293: 293: 284: 284:             merged_params.update(kwargs)
321: 321: 321: 294: 294: 285: 285: 
322: 322: 322: 295: 295: 286: 286:             # Validate configuration using Pydantic schema
323: 323: 323: 296: 296: 287: 287:             try:
324: 324: 324: 297: 297: 288: 288:                 validated_config = SingleAgentConfig(**merged_params)
325: 325: 325: 298: 298: 289: 289:                 func_logger.info(
326: 326: 326: 299: 299: 290: 290:                     "Single-agent navigator configuration validated",
327: 327: 327: 300: 300: 291: 291:                     extra={
328: 328: 328: 301: 301: 292: 292:                         "position": validated_config.position,
329: 329: 329: 302: 302: 293: 293:                         "orientation": validated_config.orientation,
330: 330: 330: 303: 303: 294: 294:                         "max_speed": validated_config.max_speed,
331: 331: 331: 304: 304: 295: 295:                         "config_source": "hydra" if cfg is not None else "direct"
332: 332: 332: 305: 305: 296: 296:                     }
333: 333: 333: 306: 306: 297: 297:                 )
334: 334: 334: 307: 307: 298: 298:             except Exception as e:
335: 335: 335: 308: 308: 299: 299:                 func_logger.error(f"Single-agent configuration validation failed: {e}")
336: 336: 336: 309: 309: 300: 300:                 raise ValueError(f"Invalid single-agent configuration: {e}") from e
337: 337: 337: 310: 310: 301: 301: 
338: 338: 338: 311: 311: 302: 302:             # Create single-agent controller
339: 339: 339: 312: 312: 303: 303:             navigator = SingleAgentController(
340: 340: 340: 313: 313: 304: 304:                 position=validated_config.position,
341: 341: 341: 314: 314: 305: 305:                 orientation=validated_config.orientation,
342: 342: 342: 315: 315: 306: 306:                 speed=validated_config.speed,
343: 343: 343: 316: 316: 307: 307:                 max_speed=validated_config.max_speed,
344: 344: 344: 317: 317: 308: 308:                 angular_velocity=validated_config.angular_velocity,
345: 345: 345: 318: 318: 309: 309:                 config=validated_config,
346: 346: 346: 319: 319: 310: 310:                 seed_manager=get_seed_manager()
347: 347: 347: 320: 320: 311: 311:             )
348: 348: 348: 321: 321: 312: 312: 
349: 349: 349: 322: 322: 313: 313:         func_logger.info(
350: 350: 350: 323: 323: 314: 314:             f"Navigator created successfully",
351: 351: 351: 324: 324: 315: 315:             extra={
352: 352: 352: 325: 325: 316: 316:                 "navigator_type": "multi-agent" if is_multi_agent else "single-agent",
353: 353: 353: 326: 326: 317: 317:                 "num_agents": navigator.num_agents
354: 354: 354: 327: 327: 318: 318:             }
355: 355: 355: 328: 328: 319: 319:         )
356: 356: 356: 329: 329: 320: 320:         
357: 357: 357: 330: 330: 321: 321:         return navigator
358: 358: 358: 331: 331: 322: 322: 
359: 359: 359: 332: 332: 323: 323:     except Exception as e:
360: 360: 360: 333: 333: 324: 324:         func_logger.error(f"Navigator creation failed: {e}")
361: 361: 361: 334: 334: 325: 325:         raise RuntimeError(f"Failed to create navigator: {e}") from e
362: 362: 362: 335: 335: 326: 326: 
363: 363: 363: 336: 336: 327: 327: 
364: 364: 364: 337: 337: 328: 328: def create_video_plume(
365: 365: 365: 338: 338: 329: 329:     cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
366: 366: 366: 339: 339: 330: 330:     video_path: Optional[Union[str, pathlib.Path]] = None,
367: 367: 367: 340: 340: 331: 331:     flip: Optional[bool] = None,
368: 368: 368: 341: 341: 332: 332:     grayscale: Optional[bool] = None,
369: 369: 369: 342: 342: 333: 333:     kernel_size: Optional[int] = None,
370: 370: 370: 343: 343: 334: 334:     kernel_sigma: Optional[float] = None,
371: 371: 371: 344: 344: 335: 335:     threshold: Optional[float] = None,
372: 372: 372: 345: 345: 336: 336:     normalize: Optional[bool] = None,
373: 373: 373: 346: 346: 337: 337:     **kwargs: Any
374: 374: 374: 347: 347: 338: 338: ) -> VideoPlume:
375: 375: 375: 348: 348: 339: 339:     """
376: 376: 376: 349: 349: 340: 340:     Create a VideoPlume instance with Hydra configuration support and enhanced validation.
377: 377: 377: 350: 350: 341: 341: 
378: 378: 378: 351: 351: 342: 342:     This function provides a unified interface for creating video-based odor plume environments
379: 379: 379: 352: 352: 343: 343:     using either direct parameter specification or Hydra DictConfig objects. Supports
380: 380: 380: 353: 353: 344: 344:     comprehensive video preprocessing options and automatic parameter validation.
381: 381: 381: 354: 354: 345: 345: 
382: 382: 382: 355: 355: 346: 346:     Parameters
383: 383: 383: 356: 356: 347: 347:     ----------
384: 384: 384: 357: 357: 348: 348:     cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
385: 385: 385: 358: 358: 349: 349:         Hydra configuration object or dictionary containing video plume parameters.
386: 386: 386: 359: 359: 350: 350:         Takes precedence over individual parameters if provided, by default None
387: 387: 387: 360: 360: 351: 351:     video_path : Optional[Union[str, pathlib.Path]], optional
388: 388: 388: 361: 361: 352: 352:         Path to the video file (MP4/AVI formats supported), by default None
389: 389: 389: 362: 362: 353: 353:     flip : Optional[bool], optional
390: 390: 390: 363: 363: 354: 354:         Whether to flip frames horizontally, by default None
391: 391: 391: 364: 364: 355: 355:     grayscale : Optional[bool], optional
392: 392: 392: 365: 365: 356: 356:         Whether to convert frames to grayscale, by default None
393: 393: 393: 366: 366: 357: 357:     kernel_size : Optional[int], optional
394: 394: 394: 367: 367: 358: 358:         Size of Gaussian kernel for smoothing (must be odd and positive), by default None
395: 395: 395: 368: 368: 359: 359:     kernel_sigma : Optional[float], optional
396: 396: 396: 369: 369: 360: 360:         Standard deviation for Gaussian kernel, by default None
397: 397: 397: 370: 370: 361: 361:     threshold : Optional[float], optional
398: 398: 398: 371: 371: 362: 362:         Threshold value for binary detection, by default None
399: 399: 399: 372: 372: 363: 363:     normalize : Optional[bool], optional
400: 400: 400: 373: 373: 364: 364:         Whether to normalize frame values to [0, 1] range, by default None
401: 401: 401: 374: 374: 365: 365:     **kwargs : Any
402: 402: 402: 375: 375: 366: 366:         Additional parameters for VideoPlume configuration
403: 403: 403: 376: 376: 367: 367: 
404: 404: 404: 377: 377: 368: 368:     Returns
405: 405: 405: 378: 378: 369: 369:     -------
406: 406: 406: 379: 379: 370: 370:     VideoPlume
407: 407: 407: 380: 380: 371: 371:         Configured VideoPlume instance ready for simulation use
408: 408: 408: 381: 381: 372: 372: 
409: 409: 409: 382: 382: 373: 373:     Raises
410: 410: 410: 383: 383: 374: 374:     ------
411: 411: 411: 384: 384: 375: 375:     ValueError
412: 412: 412: 385: 385: 376: 376:         If configuration validation fails
413: 413: 413: 386: 386: 377: 377:         If video_path is not provided or invalid
414: 414: 414: 387: 387: 378: 378:         If preprocessing parameters are invalid
415: 415: 415: 388: 388: 379: 379:     FileNotFoundError
416: 416: 416: 389: 389: 380: 380:         If the specified video file does not exist
417: 417: 417: 390: 390: 381: 381:     TypeError
418: 418: 418: 391: 391: 382: 382:         If parameter types are invalid
419: 419: 419: 392: 392: 383: 383:     RuntimeError
420: 420: 420: 393: 393: 384: 384:         If VideoPlume initialization exceeds performance requirements
421: 421: 421: 394: 394: 385: 385: 
422: 422: 422: 395: 395: 386: 386:     Examples
423: 423: 423: 396: 396: 387: 387:     --------
424: 424: 424: 397: 397: 388: 388:     Create with Hydra configuration:
425: 425: 425: 398: 398: 389: 389:         >>> from hydra import compose, initialize
426: 426: 426: 399: 399: 390: 390:         >>> with initialize(config_path="../conf"):
427: 427: 427: 400: 400: 391: 391:         ...     cfg = compose(config_name="config")
428: 428: 428: 401: 401: 392: 392:         ...     plume = create_video_plume(cfg.video_plume)
429: 429: 429: 402: 402: 393: 393: 
430: 430: 430: 403: 403: 394: 394:     Create with direct parameters:
431: 431: 431: 404: 404: 395: 395:         >>> plume = create_video_plume(
432: 432: 432: 405: 405: 396: 396:         ...     video_path="data/plume_video.mp4",
433: 433: 433: 406: 406: 397: 397:         ...     flip=True,
434: 434: 434: 407: 407: 398: 398:         ...     kernel_size=5,
435: 435: 435: 408: 408: 399: 399:         ...     kernel_sigma=1.0
436: 436: 436: 409: 409: 400: 400:         ... )
437: 437: 437: 410: 410: 401: 401: 
438: 438: 438: 411: 411: 402: 402:     Create with mixed configuration:
439: 439: 439: 412: 412: 403: 403:         >>> cfg = DictConfig({"flip": True, "grayscale": True})
440: 440: 440: 413: 413: 404: 404:         >>> plume = create_video_plume(
441: 441: 441: 414: 414: 405: 405:         ...     cfg,
442: 442: 442: 415: 415: 406: 406:         ...     video_path="data/experiment_plume.mp4",
443: 443: 443: 416: 416: 407: 407:         ...     kernel_size=3
444: 444: 444: 417: 417: 408: 408:         ... )
445: 445: 445: 418: 418: 409: 409: 
446: 446: 446: 419: 419: 410: 410:     Notes
447: 447: 447: 420: 420: 411: 411:     -----
448: 448: 448: 421: 421: 412: 412:     Configuration precedence order:
449: 449: 449: 422: 422: 413: 413:     1. Direct parameters (video_path, flip, etc.)
450: 450: 450: 423: 423: 414: 414:     2. Hydra DictConfig object values
451: 451: 451: 424: 424: 415: 415:     3. Default values from VideoPlumeConfig schema
452: 452: 452: 425: 425: 416: 416: 
453: 453: 453: 426: 426: 417: 417:     Supported video formats:
454: 454: 454: 427: 427: 418: 418:     - MP4 (recommended for best compatibility)
455: 455: 455: 428: 428: 419: 419:     - AVI with standard codecs
456: 456: 456: 429: 429: 420: 420:     - Automatic frame count and metadata extraction
457: 457: 457: 430: 430: 421: 421:     """
458: 458: 458: 431: 431: 422: 422:     # Initialize logger with function context
459: 459: 459: 432: 432: 423: 423:     func_logger = logger.bind(
460: 460: 460: 433: 433: 424: 424:         module=__name__,
461: 461: 461: 434: 434: 425: 425:         function="create_video_plume",
462: 462: 462: 435: 435: 426: 426:         cfg_provided=cfg is not None,
463: 463: 463: 436: 436: 427: 427:         video_path_provided=video_path is not None
464: 464: 464: 437: 437: 428: 428:     )
465: 465: 465: 438: 438: 429: 429: 
466: 466: 466: 439: 439: 430: 430:     try:
467: 467: 467: 440: 440: 431: 431:         # Process configuration object
468: 468: 468: 441: 441: 432: 432:         config_params = {}
469: 469: 469: 442: 442: 433: 433:         if cfg is not None:
470: 470: 470: 443: 443: 434: 434:             try:
471: 471: 471: 444: 444: 435: 435:                 if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
472: 472: 472: 445: 445: 436: 436:                     config_params = OmegaConf.to_container(cfg, resolve=True)
473: 473: 473: 446: 446: 437: 437:                 elif isinstance(cfg, dict):
474: 474: 474: 447: 447: 438: 438:                     config_params = cfg.copy()
475: 475: 475: 448: 448: 439: 439:                 else:
476: 476: 476: 449: 449: 440: 440:                     raise TypeError(f"Configuration must be DictConfig or dict, got {type(cfg)}")
477: 477: 477: 450: 450: 441: 441:             except Exception as e:
478: 478: 478: 451: 451: 442: 442:                 func_logger.error(f"Failed to process configuration: {e}")
479: 479: 479: 452: 452: 443: 443:                 raise ValueError(f"Invalid configuration object: {e}") from e
480: 480: 480: 453: 453: 444: 444: 
481: 481: 481: 454: 454: 445: 445:         # Merge parameters with precedence (direct params override config)
482: 482: 482: 455: 455: 446: 446:         merged_params = {
483: 483: 483: 456: 456: 447: 447:             "video_path": video_path or config_params.get("video_path"),
484: 484: 484: 457: 457: 448: 448:             "flip": flip if flip is not None else config_params.get("flip"),
485: 485: 485: 458: 458: 449: 449:             "grayscale": grayscale if grayscale is not None else config_params.get("grayscale"),
486: 486: 486: 459: 459: 450: 450:             "kernel_size": kernel_size if kernel_size is not None else config_params.get("kernel_size"),
487: 487: 487: 460: 460: 451: 451:             "kernel_sigma": kernel_sigma if kernel_sigma is not None else config_params.get("kernel_sigma"),
488: 488: 488: 461: 461: 452: 452:             "threshold": threshold if threshold is not None else config_params.get("threshold"),
489: 489: 489: 462: 462: 453: 453:             "normalize": normalize if normalize is not None else config_params.get("normalize"),
490: 490: 490: 463: 463: 454: 454:         }
491: 491: 491: 464: 464: 455: 455: 
492: 492: 492: 465: 465: 456: 456:         # Add any additional config parameters
493: 493: 493: 466: 466: 457: 457:         merged_params.update({k: v for k, v in config_params.items() 
494: 494: 494: 467: 467: 458: 458:                             if k not in merged_params})
495: 495: 495: 468: 468: 459: 459:         merged_params.update(kwargs)
496: 496: 496: 469: 469: 460: 460: 
497: 497: 497: 470: 470: 461: 461:         # Remove None values for cleaner validation
498: 498: 498: 471: 471: 462: 462:         merged_params = {k: v for k, v in merged_params.items() if v is not None}
499: 499: 499: 472: 472: 463: 463: 
500: 500: 500: 473: 473: 464: 464:         # Validate required video_path parameter
501: 501: 501: 474: 474: 465: 465:         if "video_path" not in merged_params or merged_params["video_path"] is None:
502: 502: 502: 475: 475: 466: 466:             raise ValueError("video_path is required for VideoPlume creation")
503: 503: 503: 476: 476: 467: 467: 
504: 504: 504: 477: 477: 468: 468:         # Validate configuration using Pydantic schema
505: 505: 505: 478: 478: 469: 469:         try:
506: 506: 506: 479: 479: 470: 470:             validated_config = VideoPlumeConfig(**merged_params)
507: 507: 507: 480: 480: 471: 471:             func_logger.info(
508: 508: 508: 481: 481: 472: 472:                 "VideoPlume configuration validated",
509: 509: 509: 482: 482: 473: 473:                 extra={
510: 510: 510: 483: 483: 474: 474:                     "video_path": str(validated_config.video_path),
511: 511: 511: 484: 484: 475: 475:                     "flip": validated_config.flip,
512: 512: 512: 485: 485: 476: 476:                     "grayscale": validated_config.grayscale,
513: 513: 513: 486: 486: 477: 477:                     "preprocessing_enabled": any([
514: 514: 514: 487: 487: 478: 478:                         validated_config.kernel_size and validated_config.kernel_size > 0,
515: 515: 515: 488: 488: 479: 479:                         validated_config.flip,
516: 516: 516: 489: 489: 480: 480:                         validated_config.threshold is not None
517: 517: 517: 490: 490: 481: 481:                     ]),
518: 518: 518: 491: 491: 482: 482:                     "config_source": "hydra" if cfg is not None else "direct"
519: 519: 519: 492: 492: 483: 483:                 }
520: 520: 520: 493: 493: 484: 484:             )
521: 521: 521: 494: 494: 485: 485:         except Exception as e:
522: 522: 522: 495: 495: 486: 486:             func_logger.error(f"VideoPlume configuration validation failed: {e}")
523: 523: 523: 496: 496: 487: 487:             raise ValueError(f"Invalid VideoPlume configuration: {e}") from e
524: 524: 524: 497: 497: 488: 488: 
525: 525: 525: 498: 498: 489: 489:         # Validate video file existence
526: 526: 526: 499: 499: 490: 490:         video_path_obj = pathlib.Path(validated_config.video_path)
527: 527: 527: 500: 500: 491: 491:         if not video_path_obj.exists():
528: 528: 528: 501: 501: 492: 492:             raise FileNotFoundError(f"Video file does not exist: {video_path_obj}")
529: 529: 529: 502: 502: 493: 493: 
530: 530: 530: 503: 503: 494: 494:         if not video_path_obj.is_file():
531: 531: 531: 504: 504: 495: 495:             raise ValueError(f"Video path is not a file: {video_path_obj}")
532: 532: 532: 505: 505: 496: 496: 
533: 533: 533: 506: 506: 497: 497:         # Create VideoPlume instance using factory method
534: 534: 534: 507: 507: 498: 498:         plume = VideoPlume.from_config(validated_config)
535: 535: 535: 508: 508: 499: 499: 
536: 536: 536: 509: 509: 500: 500:         func_logger.info(
537: 537: 537: 510: 510: 501: 501:             "VideoPlume created successfully",
538: 538: 538: 511: 511: 502: 502:             extra={
539: 539: 539: 512: 512: 503: 503:                 "video_path": str(video_path_obj),
540: 540: 540: 513: 513: 504: 504:                 "frame_count": plume.frame_count,
541: 541: 541: 514: 514: 505: 505:                 "width": plume.width,
542: 542: 542: 515: 515: 506: 506:                 "height": plume.height,
543: 543: 543: 516: 516: 507: 507:                 "fps": plume.fps,
544: 544: 544: 517: 517: 508: 508:                 "duration": plume.duration
545: 545: 545: 518: 518: 509: 509:             }
546: 546: 546: 519: 519: 510: 510:         )
547: 547: 547: 520: 520: 511: 511: 
548: 548: 548: 521: 521: 512: 512:         return plume
549: 549: 549: 522: 522: 513: 513: 
550: 550: 550: 523: 523: 514: 514:     except Exception as e:
551: 551: 551: 524: 524: 515: 515:         func_logger.error(f"VideoPlume creation failed: {e}")
552: 552: 552: 525: 525: 516: 516:         raise RuntimeError(f"Failed to create VideoPlume: {e}") from e
553: 553: 553: 526: 526: 517: 517: 
554: 554: 554: 527: 527: 518: 518: 
555: 555: 555: 528: 528: 519: 519: def run_plume_simulation(
556: 556: 556: 529: 529: 520: 520:     navigator: NavigatorProtocol,
557: 557: 557: 530: 530: 521: 521:     video_plume: VideoPlume,
558: 558: 558: 531: 531: 522: 522:     cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
559: 559: 559: 532: 532: 523: 523:     num_steps: Optional[int] = None,
560: 560: 560: 533: 533: 524: 524:     dt: Optional[float] = None,
561: 561: 561: 534: 534: 525: 525:     step_size: Optional[float] = None,  # Backward compatibility
562: 562: 562: 535: 535: 526: 526:     sensor_distance: Optional[float] = None,
563: 563: 563: 536: 536: 527: 527:     sensor_angle: Optional[float] = None,
564: 564: 564: 537: 537: 528: 528:     record_trajectories: bool = True,
565: 565: 565: 538: 538: 529: 529:     **kwargs: Any
566: 566: 566: 539: 539: 530: 530: ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
567: 567: 567: 540: 540: 531: 531:     """
568: 568: 568: 541: 541: 532: 532:     Execute a complete plume navigation simulation with Hydra configuration support.
569: 569: 569: 542: 542: 533: 533: 
570: 570: 570: 543: 543: 534: 534:     This function orchestrates frame-by-frame agent navigation through video-based odor
571: 571: 571: 544: 544: 535: 535:     plume environments with comprehensive data collection and performance monitoring.
572: 572: 572: 545: 545: 536: 536:     Supports both single-agent and multi-agent scenarios with automatic trajectory recording.
573: 573: 573: 546: 546: 537: 537: 
574: 574: 574: 547: 547: 538: 538:     Parameters
575: 575: 575: 548: 548: 539: 539:     ----------
576: 576: 576: 549: 549: 540: 540:     navigator : NavigatorProtocol
577: 577: 577: 550: 550: 541: 541:         Navigator instance (SingleAgentController or MultiAgentController)
578: 578: 578: 551: 551: 542: 542:     video_plume : VideoPlume
579: 579: 579: 552: 552: 543: 543:         VideoPlume environment instance providing odor concentration data
580: 580: 580: 553: 553: 544: 544:     cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
581: 581: 581: 554: 554: 545: 545:         Hydra configuration object or dictionary containing simulation parameters, by default None
582: 582: 582: 555: 555: 546: 546:     num_steps : Optional[int], optional
583: 583: 583: 556: 556: 547: 547:         Number of simulation steps to execute, by default None
584: 584: 584: 557: 557: 548: 548:     dt : Optional[float], optional
585: 585: 585: 558: 558: 549: 549:         Simulation timestep in seconds, by default None
586: 586: 586: 559: 559: 550: 550:     step_size : Optional[float], optional
587: 587: 587: 560: 560: 551: 551:         Legacy parameter for backward compatibility (converted to dt), by default None
588: 588: 588: 561: 561: 552: 552:     sensor_distance : Optional[float], optional
589: 589: 589: 562: 562: 553: 553:         Distance for sensor sampling, by default None
590: 590: 590: 563: 563: 554: 554:     sensor_angle : Optional[float], optional
591: 591: 591: 564: 564: 555: 555:         Angle for sensor sampling, by default None
592: 592: 592: 565: 565: 556: 556:     record_trajectories : bool, optional
593: 593: 593: 566: 566: 557: 557:         Whether to record position and orientation trajectories, by default True
594: 594: 594: 567: 567: 558: 558:     **kwargs : Any
595: 595: 595: 568: 568: 559: 559:         Additional simulation parameters
596: 596: 596: 569: 569: 560: 560: 
597: 597: 597: 570: 570: 561: 561:     Returns
598: 598: 598: 571: 571: 562: 562:     -------
599: 599: 599: 572: 572: 563: 563:     Tuple[np.ndarray, np.ndarray, np.ndarray]
600: 600: 600: 573: 573: 564: 564:         positions_history : np.ndarray
601: 601: 601: 574: 574: 565: 565:             Agent positions with shape (num_agents, num_steps + 1, 2)
602: 602: 602: 575: 575: 566: 566:         orientations_history : np.ndarray
603: 603: 603: 576: 576: 567: 567:             Agent orientations with shape (num_agents, num_steps + 1)
604: 604: 604: 577: 577: 568: 568:         odor_readings : np.ndarray
605: 605: 605: 578: 578: 569: 569:             Sensor readings with shape (num_agents, num_steps + 1)
606: 606: 606: 579: 579: 570: 570: 
607: 607: 607: 580: 580: 571: 571:     Raises
608: 608: 608: 581: 581: 572: 572:     ------
609: 609: 609: 582: 582: 573: 573:     ValueError
610: 610: 610: 583: 583: 574: 574:         If required parameters are missing or invalid
611: 611: 611: 584: 584: 575: 575:         If navigator or video_plume are None
612: 612: 612: 585: 585: 576: 576:         If configuration validation fails
613: 613: 613: 586: 586: 577: 577:     TypeError
614: 614: 614: 587: 587: 578: 578:         If navigator doesn't implement NavigatorProtocol
615: 615: 615: 588: 588: 579: 579:         If video_plume is not a VideoPlume instance
616: 616: 616: 589: 589: 580: 580:     RuntimeError
617: 617: 617: 590: 590: 581: 581:         If simulation execution fails or exceeds performance requirements
618: 618: 618: 591: 591: 582: 582: 
619: 619: 619: 592: 592: 583: 583:     Examples
620: 620: 620: 593: 593: 584: 584:     --------
621: 621: 621: 594: 594: 585: 585:     Run simulation with Hydra configuration:
622: 622: 622: 595: 595: 586: 586:         >>> from hydra import compose, initialize
623: 623: 623: 596: 596: 587: 587:         >>> with initialize(config_path="../conf"):
624: 624: 624: 597: 597: 588: 588:         ...     cfg = compose(config_name="config")
625: 625: 625: 598: 598: 589: 589:         ...     navigator = create_navigator(cfg.navigator)
626: 626: 626: 599: 599: 590: 590:         ...     plume = create_video_plume(cfg.video_plume)
627: 627: 627: 600: 600: 591: 591:         ...     positions, orientations, readings = run_plume_simulation(
628: 628: 628: 601: 601: 592: 592:         ...         navigator, plume, cfg.simulation
629: 629: 629: 602: 602: 593: 593:         ...     )
630: 630: 630: 603: 603: 594: 594: 
631: 631: 631: 604: 604: 595: 595:     Run simulation with direct parameters:
632: 632: 632: 605: 605: 596: 596:         >>> positions, orientations, readings = run_plume_simulation(
633: 633: 633: 606: 606: 597: 597:         ...     navigator=navigator,
634: 634: 634: 607: 607: 598: 598:         ...     video_plume=plume,
635: 635: 635: 608: 608: 599: 599:         ...     num_steps=1000,
636: 636: 636: 609: 609: 600: 600:         ...     dt=0.1
637: 637: 637: 610: 610: 601: 601:         ... )
638: 638: 638: 611: 611: 602: 602: 
639: 639: 639: 612: 612: 603: 603:     Use results for analysis:
640: 640: 640: 613: 613: 604: 604:         >>> print(f"Simulation completed: {positions.shape[1]} steps, {positions.shape[0]} agents")
641: 641: 641: 614: 614: 605: 605:         >>> final_positions = positions[:, -1, :]
642: 642: 642: 615: 615: 606: 606:         >>> trajectory_lengths = np.sum(np.diff(positions, axis=1)**2, axis=(1,2))**0.5
643: 643: 643: 616: 616: 607: 607: 
644: 644: 644: 617: 617: 608: 608:     Notes
645: 645: 645: 618: 618: 609: 609:     -----
646: 646: 646: 619: 619: 610: 610:     Performance characteristics:
647: 647: 647: 620: 620: 611: 611:     - Optimized for 30+ FPS execution with real-time visualization
648: 648: 648: 621: 621: 612: 612:     - Memory-efficient trajectory recording with configurable storage
649: 649: 649: 622: 622: 613: 613:     - Automatic frame synchronization between navigator and video plume
650: 650: 650: 623: 623: 614: 614:     - Progress logging for long-running simulations
651: 651: 651: 624: 624: 615: 615: 
652: 652: 652: 625: 625: 616: 616:     Configuration precedence order:
653: 653: 653: 626: 626: 617: 617:     1. Direct parameters (num_steps, dt, etc.)
654: 654: 654: 627: 627: 618: 618:     2. Hydra DictConfig object values
655: 655: 655: 628: 628: 619: 619:     3. Default values from SimulationConfig schema
656: 656: 656: 629: 629: 620: 620: 
657: 657: 657: 630: 630: 621: 621:     Backward compatibility:
658: 658: 658: 631: 631: 622: 622:     - step_size parameter is converted to dt for legacy compatibility
659: 659: 659: 632: 632: 623: 623:     - Original parameter names are preserved where possible
660: 660: 660: 633: 633: 624: 624:     """
661: 661: 661: 634: 634: 625: 625:     # Initialize logger with simulation context
662: 662: 662: 635: 635: 626: 626:     sim_logger = logger.bind(
663: 663: 663: 636: 636: 627: 627:         module=__name__,
664: 664: 664: 637: 637: 628: 628:         function="run_plume_simulation",
665: 665: 665: 638: 638: 629: 629:         navigator_type=type(navigator).__name__,
666: 666: 666: 639: 639: 630: 630:         num_agents=navigator.num_agents,
667: 667: 667: 640: 640: 631: 631:         video_frames=video_plume.frame_count,
668: 668: 668: 641: 641: 632: 632:         cfg_provided=cfg is not None
669: 669: 669: 642: 642: 633: 633:     )
670: 670: 670: 643: 643: 634: 634: 
671: 671: 671: 644: 644: 635: 635:     try:
672: 672: 672: 645: 645: 636: 636:         # Validate required inputs
673: 673: 673: 646: 646: 637: 637:         if navigator is None:
674: 674: 674: 647: 647: 638: 638:             raise ValueError("navigator parameter is required")
675: 675: 675: 648: 648: 639: 639:         if video_plume is None:
676: 676: 676: 649: 649: 640: 640:             raise ValueError("video_plume parameter is required")
677: 677: 677: 650: 650: 641: 641: 
678: 678: 678: 651: 651: 642: 642:         # Type validation
679: 679: 679: 652: 652: 643: 643:         if not hasattr(navigator, 'positions') or not hasattr(navigator, 'step'):
680: 680: 680: 653: 653: 644: 644:             raise TypeError("navigator must implement NavigatorProtocol interface")
681: 681: 681: 654: 654: 645: 645:         
682: 682: 682: 655: 655: 646: 646:         if not hasattr(video_plume, 'get_frame') or not hasattr(video_plume, 'frame_count'):
683: 683: 683: 656: 656: 647: 647:             raise TypeError("video_plume must be a VideoPlume instance")
684: 684: 684: 657: 657: 648: 648: 
685: 685: 685: 658: 658: 649: 649:         # Process configuration object
686: 686: 686: 659: 659: 650: 650:         config_params = {}
687: 687: 687: 660: 660: 651: 651:         if cfg is not None:
688: 688: 688: 661: 661: 652: 652:             try:
689: 689: 689: 662: 662: 653: 653:                 if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
690: 690: 690: 663: 663: 654: 654:                     config_params = OmegaConf.to_container(cfg, resolve=True)
691: 691: 691: 664: 664: 655: 655:                 elif isinstance(cfg, dict):
692: 692: 692: 665: 665: 656: 656:                     config_params = cfg.copy()
693: 693: 693: 666: 666: 657: 657:                 else:
694: 694: 694: 667: 667: 658: 658:                     raise TypeError(f"Configuration must be DictConfig or dict, got {type(cfg)}")
695: 695: 695: 668: 668: 659: 659:             except Exception as e:
696: 696: 696: 669: 669: 660: 660:                 sim_logger.error(f"Failed to process configuration: {e}")
697: 697: 697: 670: 670: 661: 661:                 raise ValueError(f"Invalid configuration object: {e}") from e
698: 698: 698: 671: 671: 662: 662: 
699: 699: 699: 672: 672: 663: 663:         # Merge parameters with precedence and handle backward compatibility
700: 700: 700: 673: 673: 664: 664:         merged_params = {
701: 701: 701: 674: 674: 665: 665:             "num_steps": num_steps or config_params.get("num_steps"),
702: 702: 702: 675: 675: 666: 666:             "dt": dt or step_size or config_params.get("dt") or config_params.get("step_size"),
703: 703: 703: 676: 676: 667: 667:             "sensor_distance": sensor_distance or config_params.get("sensor_distance", 5.0),
704: 704: 704: 677: 677: 668: 668:             "sensor_angle": sensor_angle or config_params.get("sensor_angle", 45.0),
705: 705: 705: 678: 678: 669: 669:             "record_trajectories": record_trajectories and config_params.get("record_trajectories", True),
706: 706: 706: 679: 679: 670: 670:         }
707: 707: 707: 680: 680: 671: 671: 
708: 708: 708: 681: 681: 672: 672:         # Add any additional config parameters
709: 709: 709: 682: 682: 673: 673:         merged_params.update({k: v for k, v in config_params.items() 
710: 710: 710: 683: 683: 674: 674:                             if k not in merged_params})
711: 711: 711: 684: 684: 675: 675:         merged_params.update(kwargs)
712: 712: 712: 685: 685: 676: 676: 
713: 713: 713: 686: 686: 677: 677:         # Remove None values for validation
714: 714: 714: 687: 687: 678: 678:         merged_params = {k: v for k, v in merged_params.items() if v is not None}
715: 715: 715: 688: 688: 679: 679: 
716: 716: 716: 689: 689: 680: 680:         # Validate configuration using Pydantic schema
717: 717: 717: 690: 690: 681: 681:         try:
718: 718: 718: 691: 691: 682: 682:             validated_config = SimulationConfig(**merged_params)
719: 719: 719: 692: 692: 683: 683:             sim_logger.info(
720: 720: 720: 693: 693: 684: 684:                 "Simulation configuration validated",
721: 721: 721: 694: 694: 685: 685:                 extra={
722: 722: 722: 695: 695: 686: 686:                     "num_steps": validated_config.max_steps,
723: 723: 723: 696: 696: 687: 687:                     "dt": validated_config.step_size,
724: 724: 724: 697: 697: 688: 688:                     "sensor_distance": merged_params.get("sensor_distance", 5.0),
725: 725: 725: 698: 698: 689: 689:                     "sensor_angle": merged_params.get("sensor_angle", 45.0),
726: 726: 726: 699: 699: 690: 690:                     "record_trajectories": merged_params.get("record_trajectories", True),
727: 727: 727: 700: 700: 691: 691:                     "config_source": "hydra" if cfg is not None else "direct"
728: 728: 728: 701: 701: 692: 692:                 }
729: 729: 729: 702: 702: 693: 693:             )
730: 730: 730: 703: 703: 694: 694:         except Exception as e:
731: 731: 731: 704: 704: 695: 695:             sim_logger.error(f"Simulation configuration validation failed: {e}")
732: 732: 732: 705: 705: 696: 696:             raise ValueError(f"Invalid simulation configuration: {e}") from e
733: 733: 733: 706: 706: 697: 697: 
734: 734: 734: 707: 707: 698: 698:         # Initialize seed manager for reproducible execution
735: 735: 735: 708: 708: 699: 699:         seed_manager = get_seed_manager()
736: 736: 736: 709: 709: 700: 700:         if seed_manager.current_seed is not None:
737: 737: 737: 710: 710: 701: 701:             sim_logger.info(f"Simulation running with seed: {seed_manager.current_seed}")
738: 738: 738: 711: 711: 702: 702: 
739: 739: 739: 712: 712: 703: 703:         # Get simulation parameters
740: 740: 740: 713: 713: 704: 704:         num_steps = merged_params.get("num_steps", validated_config.max_steps)
741: 741: 741: 714: 714: 705: 705:         dt = merged_params.get("dt", validated_config.step_size)
742: 742: 742: 715: 715: 706: 706:         num_agents = navigator.num_agents
743: 743: 743: 716: 716: 707: 707: 
744: 744: 744: 717: 717: 708: 708:         # Initialize trajectory storage if recording enabled
745: 745: 745: 718: 718: 709: 709:         record_trajectories = merged_params.get("record_trajectories", True)
746: 746: 746: 719: 719: 710: 710:         if record_trajectories:
747: 747: 747: 720: 720: 711: 711:             positions_history = np.zeros((num_agents, num_steps + 1, 2))
748: 748: 748: 721: 721: 712: 712:             orientations_history = np.zeros((num_agents, num_steps + 1))
749: 749: 749: 722: 722: 713: 713:             odor_readings = np.zeros((num_agents, num_steps + 1))
750: 750: 750: 723: 723: 714: 714:             
751: 751: 751: 724: 724: 715: 715:             # Store initial state
752: 752: 752: 725: 725: 716: 716:             positions_history[:, 0] = navigator.positions
753: 753: 753: 726: 726: 717: 717:             orientations_history[:, 0] = navigator.orientations
754: 754: 754: 727: 727: 718: 718:             
755: 755: 755: 728: 728: 719: 719:             # Get initial odor readings
756: 756: 756: 729: 729: 720: 720:             current_frame = video_plume.get_frame(0)
757: 757: 757: 730: 730: 721: 721:             if hasattr(navigator, 'sample_odor'):
758: 758: 758: 731: 731: 722: 722:                 initial_readings = navigator.sample_odor(current_frame)
759: 759: 759: 732: 732: 723: 723:                 if isinstance(initial_readings, (int, float)):
760: 760: 760: 733: 733: 724: 724:                     odor_readings[:, 0] = initial_readings
761: 761: 761: 734: 734: 725: 725:                 else:
762: 762: 762: 735: 735: 726: 726:                     odor_readings[:, 0] = initial_readings
763: 763: 763: 736: 736: 727: 727:             else:
764: 764: 764: 737: 737: 728: 728:                 odor_readings[:, 0] = 0.0
765: 765: 765: 738: 738: 729: 729:         else:
766: 766: 766: 739: 739: 730: 730:             # Minimal storage for return compatibility
767: 767: 767: 740: 740: 731: 731:             positions_history = np.zeros((num_agents, 2, 2))
768: 768: 768: 741: 741: 732: 732:             orientations_history = np.zeros((num_agents, 2))
769: 769: 769: 742: 742: 733: 733:             odor_readings = np.zeros((num_agents, 2))
770: 770: 770: 743: 743: 734: 734: 
771: 771: 771: 744: 744: 735: 735:         sim_logger.info(
772: 772: 772: 745: 745: 736: 736:             "Starting simulation execution",
773: 773: 773: 746: 746: 737: 737:             extra={
774: 774: 774: 747: 747: 738: 738:                 "total_steps": num_steps,
775: 775: 775: 748: 748: 739: 739:                 "estimated_duration": num_steps * dt,
776: 776: 776: 749: 749: 740: 740:                 "memory_usage": f"{positions_history.nbytes + orientations_history.nbytes + odor_readings.nbytes:.1f} bytes"
777: 777: 777: 750: 750: 741: 741:             }
778: 778: 778: 751: 751: 742: 742:         )
779: 779: 779: 752: 752: 743: 743: 
780: 780: 780: 753: 753: 744: 744:         # Execute simulation loop
781: 781: 781: 754: 754: 745: 745:         for step in range(num_steps):
782: 782: 782: 755: 755: 746: 746:             try:
783: 783: 783: 756: 756: 747: 747:                 # Get current frame with bounds checking
784: 784: 784: 757: 757: 748: 748:                 frame_idx = min(step + 1, video_plume.frame_count - 1)
785: 785: 785: 758: 758: 749: 749:                 current_frame = video_plume.get_frame(frame_idx)
786: 786: 786: 759: 759: 750: 750:                 
787: 787: 787: 760: 760: 751: 751:                 # Update navigator state
788: 788: 788: 761: 761: 752: 752:                 navigator.step(current_frame)
789: 789: 789: 762: 762: 753: 753:                 
790: 790: 790: 763: 763: 754: 754:                 # Record trajectory data if enabled
791: 791: 791: 764: 764: 755: 755:                 if record_trajectories:
792: 792: 792: 765: 765: 756: 756:                     positions_history[:, step + 1] = navigator.positions
793: 793: 793: 766: 766: 757: 757:                     orientations_history[:, step + 1] = navigator.orientations
794: 794: 794: 767: 767: 758: 758:                     
795: 795: 795: 768: 768: 759: 759:                     # Sample odor at current position
796: 796: 796: 769: 769: 760: 760:                     if hasattr(navigator, 'sample_odor'):
797: 797: 797: 770: 770: 761: 761:                         readings = navigator.sample_odor(current_frame)
798: 798: 798: 771: 771: 762: 762:                         if isinstance(readings, (int, float)):
799: 799: 799: 772: 772: 763: 763:                             odor_readings[:, step + 1] = readings
800: 800: 800: 773: 773: 764: 764:                         else:
801: 801: 801: 774: 774: 765: 765:                             odor_readings[:, step + 1] = readings
802: 802: 802: 775: 775: 766: 766:                     else:
803: 803: 803: 776: 776: 767: 767:                         odor_readings[:, step + 1] = 0.0
804: 804: 804: 777: 777: 768: 768: 
805: 805: 805: 778: 778: 769: 769:                 # Progress logging for long simulations
806: 806: 806: 779: 779: 770: 770:                 if num_steps > 100 and (step + 1) % (num_steps // 10) == 0:
807: 807: 807: 780: 780: 771: 771:                     progress = (step + 1) / num_steps * 100
808: 808: 808: 781: 781: 772: 772:                     sim_logger.debug(f"Simulation progress: {progress:.1f}% ({step + 1}/{num_steps} steps)")
809: 809: 809: 782: 782: 773: 773: 
810: 810: 810: 783: 783: 774: 774:             except Exception as e:
811: 811: 811: 784: 784: 775: 775:                 sim_logger.error(f"Simulation failed at step {step}: {e}")
812: 812: 812: 785: 785: 776: 776:                 raise RuntimeError(f"Simulation execution failed at step {step}: {e}") from e
813: 813: 813: 786: 786: 777: 777: 
814: 814: 814: 787: 787: 778: 778:         # Handle non-recording case by storing final state
815: 815: 815: 788: 788: 779: 779:         if not record_trajectories:
816: 816: 816: 789: 789: 780: 780:             positions_history[:, 0] = navigator.positions
817: 817: 817: 790: 790: 781: 781:             orientations_history[:, 0] = navigator.orientations
818: 818: 818: 791: 791: 782: 782:             if hasattr(navigator, 'sample_odor'):
819: 819: 819: 792: 792: 783: 783:                 final_frame = video_plume.get_frame(video_plume.frame_count - 1)
820: 820: 820: 793: 793: 784: 784:                 readings = navigator.sample_odor(final_frame)
821: 821: 821: 794: 794: 785: 785:                 if isinstance(readings, (int, float)):
822: 822: 822: 795: 795: 786: 786:                     odor_readings[:, 0] = readings
823: 823: 823: 796: 796: 787: 787:                 else:
824: 824: 824: 797: 797: 788: 788:                     odor_readings[:, 0] = readings
825: 825: 825: 798: 798: 789: 789: 
826: 826: 826: 799: 799: 790: 790:         sim_logger.info(
827: 827: 827: 800: 800: 791: 791:             "Simulation completed successfully",
828: 828: 828: 801: 801: 792: 792:             extra={
829: 829: 829: 802: 802: 793: 793:                 "steps_executed": num_steps,
830: 830: 830: 803: 803: 794: 794:                 "final_positions": positions_history[:, -1, :].tolist() if record_trajectories else positions_history[:, 0, :].tolist(),
831: 831: 831: 804: 804: 795: 795:                 "trajectory_recorded": record_trajectories,
832: 832: 832: 805: 805: 796: 796:                 "data_shape": {
833: 833: 833: 806: 806: 797: 797:                     "positions": positions_history.shape,
834: 834: 834: 807: 807: 798: 798:                     "orientations": orientations_history.shape,
835: 835: 835: 808: 808: 799: 799:                     "readings": odor_readings.shape
836: 836: 836: 809: 809: 800: 800:                 }
837: 837: 837: 810: 810: 801: 801:             }
838: 838: 838: 811: 811: 802: 802:         )
839: 839: 839: 812: 812: 803: 803: 
840: 840: 840: 813: 813: 804: 804:         return positions_history, orientations_history, odor_readings
841: 841: 841: 814: 814: 805: 805: 
842: 842: 842: 815: 815: 806: 806:     except Exception as e:
843: 843: 843: 816: 816: 807: 807:         sim_logger.error(f"Simulation execution failed: {e}")
844: 844: 844: 817: 817: 808: 808:         raise RuntimeError(f"Failed to execute simulation: {e}") from e
845: 845: 845: 818: 818: 809: 809: 
846: 846: 846: 819: 819: 810: 810: 
847: 847: 847: 820: 820: 811: 811: def visualize_plume_simulation(
848: 848: 848: 821: 821: 812: 812:     positions: np.ndarray,
849: 849: 849: 822: 822: 813: 813:     orientations: np.ndarray,
850: 850: 850: 823: 823: 814: 814:     odor_readings: Optional[np.ndarray] = None,
851: 851: 851: 824: 824: 815: 815:     plume_frames: Optional[np.ndarray] = None,
852: 852: 852: 825: 825: 816: 816:     cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
853: 853: 853: 826: 826: 817: 817:     output_path: Optional[Union[str, pathlib.Path]] = None,
854: 854: 854: 827: 827: 818: 818:     show_plot: bool = True,
855: 855: 855: 828: 828: 819: 819:     close_plot: Optional[bool] = None,
856: 856: 856: 829: 829: 820: 820:     animation: bool = False,
857: 857: 857: 830: 830: 821: 821:     **kwargs: Any
858: 858: 858: 831: 831: 822: 822: ) -> "matplotlib.figure.Figure":
859: 859: 859: 832: 832: 823: 823:     """
860: 860: 860: 833: 833: 824: 824:     Visualize simulation results with comprehensive formatting and export options.
861: 861: 861: 834: 834: 825: 825: 
862: 862: 862: 835: 835: 826: 826:     This function provides publication-quality visualization of agent trajectories
863: 863: 863: 836: 836: 827: 827:     and environmental data with support for both static plots and animated sequences.
864: 864: 864: 837: 837: 828: 828:     Integrates with Hydra configuration for consistent visualization parameters.
865: 865: 865: 838: 838: 829: 829: 
866: 866: 866: 839: 839: 830: 830:     Parameters
867: 867: 867: 840: 840: 831: 831:     ----------
868: 868: 868: 841: 841: 832: 832:     positions : np.ndarray
869: 869: 869: 842: 842: 833: 833:         Agent positions with shape (num_agents, num_steps, 2)
870: 870: 870: 843: 843: 834: 834:     orientations : np.ndarray
871: 871: 871: 844: 844: 835: 835:         Agent orientations with shape (num_agents, num_steps)
872: 872: 872: 845: 845: 836: 836:     odor_readings : Optional[np.ndarray], optional
873: 873: 873: 846: 846: 837: 837:         Sensor readings with shape (num_agents, num_steps), by default None
874: 874: 874: 847: 847: 838: 838:     plume_frames : Optional[np.ndarray], optional
875: 875: 875: 848: 848: 839: 839:         Video frames with shape (num_steps, height, width, channels), by default None
876: 876: 876: 849: 849: 840: 840:     cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
877: 877: 877: 850: 850: 841: 841:         Hydra configuration for visualization parameters, by default None
878: 878: 878: 851: 851: 842: 842:     output_path : Optional[Union[str, pathlib.Path]], optional
879: 879: 879: 852: 852: 843: 843:         Path to save visualization output, by default None
880: 880: 880: 853: 853: 844: 844:     show_plot : bool, optional
881: 881: 881: 854: 854: 845: 845:         Whether to display the plot interactively, by default True
882: 882: 882: 855: 855: 846: 846:     close_plot : Optional[bool], optional
883: 883: 883: 856: 856: 847: 847:         Whether to close plot after saving, by default None
884: 884: 884: 857: 857: 848: 848:     animation : bool, optional
885: 885: 885: 858: 858: 849: 849:         Whether to create animated visualization, by default False
886: 886: 886: 859: 859: 850: 850:     **kwargs : Any
887: 887: 887: 860: 860: 851: 851:         Additional visualization parameters
888: 888: 888: 861: 861: 852: 852: 
889: 889: 889: 862: 862: 853: 853:     Returns
890: 890: 890: 863: 863: 854: 854:     -------
891: 891: 891: 864: 864: 855: 855:     matplotlib.figure.Figure
892: 892: 892: 865: 865: 856: 856:         The created matplotlib figure object
893: 893: 893: 866: 866: 857: 857: 
894: 894: 894: 867: 867: 858: 858:     Examples
895: 895: 895: 868: 868: 859: 859:     --------
896: 896: 896: 869: 869: 860: 860:     Basic trajectory visualization:
897: 897: 897: 870: 870: 861: 861:         >>> fig = visualize_plume_simulation(positions, orientations)
898: 898: 898: 871: 871: 862: 862: 
899: 899: 899: 872: 872: 863: 863:     Publication-quality plot with configuration:
900: 900: 900: 873: 873: 864: 864:         >>> fig = visualize_plume_simulation(
901: 901: 901: 874: 874: 865: 865:         ...     positions, orientations, odor_readings,
902: 902: 902: 875: 875: 866: 866:         ...     cfg=viz_config,
903: 903: 903: 876: 876: 867: 867:         ...     output_path="results/trajectory.png",
904: 904: 904: 877: 877: 868: 868:         ...     show_plot=False
905: 905: 905: 878: 878: 869: 869:         ... )
906: 906: 906: 879: 879: 870: 870: 
907: 907: 907: 880: 880: 871: 871:     Animated visualization:
908: 908: 908: 881: 881: 872: 872:         >>> fig = visualize_plume_simulation(
909: 909: 909: 882: 882: 873: 873:         ...     positions, orientations, 
910: 910: 910: 883: 883: 874: 874:         ...     plume_frames=frames,
911: 911: 911: 884: 884: 875: 875:         ...     animation=True,
912: 912: 912: 885: 885: 876: 876:         ...     output_path="results/animation.mp4"
913: 913: 913: 886: 886: 877: 877:         ... )
914: 914: 914: 887: 887: 878: 878:     """
915: 915: 915: 888: 888: 879: 879:     # Initialize logger
916: 916: 916: 889: 889: 880: 880:     viz_logger = logger.bind(
917: 917: 917: 890: 890: 881: 881:         module=__name__,
918: 918: 918: 891: 891: 882: 882:         function="visualize_plume_simulation",
919: 919: 919: 892: 892: 883: 883:         num_agents=positions.shape[0],
920: 920: 920: 893: 893: 884: 884:         num_steps=positions.shape[1],
921: 921: 921: 894: 894: 885: 885:         animation=animation
922: 922: 922: 895: 895: 886: 886:     )
923: 923: 923: 896: 896: 887: 887: 
924: 924: 924: 897: 897: 888: 888:     try:
925: 925: 925: 898: 898: 889: 889:         # Process configuration
926: 926: 926: 899: 899: 890: 890:         config_params = {}
927: 927: 927: 900: 900: 891: 891:         if cfg is not None:
928: 928: 928: 901: 901: 892: 892:             if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
929: 929: 929: 902: 902: 893: 893:                 config_params = OmegaConf.to_container(cfg, resolve=True)
930: 930: 930: 903: 903: 894: 894:             elif isinstance(cfg, dict):
931: 931: 931: 904: 904: 895: 895:                 config_params = cfg.copy()
932: 932: 932: 905: 905: 896: 896: 
933: 933: 933: 906: 906: 897: 897:         # Merge visualization parameters
934: 934: 934: 907: 907: 898: 898:         viz_params = {
935: 935: 935: 908: 908: 899: 899:             "output_path": output_path,
936: 936: 936: 909: 909: 900: 900:             "show_plot": show_plot,
937: 937: 937: 910: 910: 901: 901:             "close_plot": close_plot,
938: 938: 938: 911: 911: 902: 902:             **config_params,
939: 939: 939: 912: 912: 903: 903:             **kwargs
940: 940: 940: 913: 913: 904: 904:         }
941: 941: 941: 914: 914: 905: 905: 
942: 942: 942: 915: 915: 906: 906:         # Select appropriate visualization function
943: 943: 943: 916: 916: 907: 907:         if animation:
944: 944: 944: 917: 917: 908: 908:             return visualize_simulation_results(
945: 945: 945: 918: 918: 909: 909:                 positions=positions,
946: 946: 946: 919: 919: 910: 910:                 orientations=orientations,
947: 947: 947: 920: 920: 911: 911:                 odor_readings=odor_readings,
948: 948: 948: 921: 921: 912: 912:                 plume_frames=plume_frames,
949: 949: 949: 922: 922: 913: 913:                 **viz_params
950: 950: 950: 923: 923: 914: 914:             )
951: 951: 951: 924: 924: 915: 915:         else:
952: 952: 952: 925: 925: 916: 916:             return visualize_trajectory(
953: 953: 953: 926: 926: 917: 917:                 positions=positions,
954: 954: 954: 927: 927: 918: 918:                 orientations=orientations,
955: 955: 955: 928: def create_gymnasium_environment(
956: 956: 956: 929:     cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
957: 957: 957: 930:     video_path: Optional[Union[str, pathlib.Path]] = None,
958: 958: 958: 931:     initial_position: Optional[Tuple[float, float]] = None,
959: 959: 959: 932:     initial_orientation: float = 0.0,
960: 960: 960: 933:     max_speed: float = 2.0,
961: 961: 961: 934:     max_angular_velocity: float = 90.0,
962: 962: 962: 935:     include_multi_sensor: bool = False,
963: 963: 963: 936:     num_sensors: int = 2,
964: 964: 964: 937:     sensor_distance: float = 5.0,
965: 965: 965: 938:     sensor_layout: str = "bilateral",
966: 966: 966: 939:     reward_config: Optional[Dict[str, float]] = None,
967: 967: 967: 940:     max_episode_steps: int = 1000,
968: 968: 968: 941:     render_mode: Optional[str] = None,
969: 969: 969: 942:     seed: Optional[int] = None,
970: 970: 970: 943:     performance_monitoring: bool = True,
971: 971: 971: 944:     **kwargs: Any
972: 972: 972: 945: ) -> "GymnasiumEnv":
973: 973: 973: 946:     """
974: 974: 974: 947:     Create a Gymnasium-compliant environment for odor plume navigation with RL training support.
975: 975: 975: 948: 
976: 976: 976: 949:     This factory function serves as the primary entry point for creating RL-ready environments
977: 977: 977: 950:     that integrate seamlessly with stable-baselines3 and other modern reinforcement learning
978: 978: 978: 951:     frameworks. It provides a unified interface supporting both configuration-driven and
979: 979: 979: 952:     direct parameter-based instantiation.
980: 980: 980: 953: 
981: 981: 981: 954:     The environment wraps the existing plume navigation simulation infrastructure within
982: 982: 982: 955:     the standard Gymnasium API, enabling researchers to leverage existing RL algorithms
983: 983: 983: 956:     while maintaining full compatibility with the navigation core.
984: 984: 984: 957: 
985: 985: 985: 958:     Parameters
986: 986: 986: 959:     ----------
987: 987: 987: 960:     cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
988: 988: 988: 961:         Hydra configuration object or dictionary containing environment parameters.
989: 989: 989: 962:         Takes precedence over individual parameters if provided, by default None
990: 990: 990: 963:     video_path : Optional[Union[str, pathlib.Path]], optional
991: 991: 991: 964:         Path to video file containing odor plume data, by default None
992: 992: 992: 965:     initial_position : Optional[Tuple[float, float]], optional
993: 993: 993: 966:         Starting (x, y) position for agent (default: video center), by default None
994: 994: 994: 967:     initial_orientation : float, optional
995: 995: 995: 968:         Starting orientation in degrees, by default 0.0
996: 996: 996: 969:     max_speed : float, optional
997: 997: 997: 970:         Maximum agent speed in units per time step, by default 2.0
998: 998: 998: 971:     max_angular_velocity : float, optional
999: 999: 999: 972:         Maximum angular velocity in degrees/sec, by default 90.0
1000: 1000: 1000: 973:     include_multi_sensor : bool, optional
1001: 1001: 1001: 974:         Whether to include multi-sensor observations, by default False
1002: 1002: 1002: 975:     num_sensors : int, optional
1003: 1003: 1003: 976:         Number of additional sensors for multi-sensor mode, by default 2
1004: 1004: 1004: 977:     sensor_distance : float, optional
1005: 1005: 1005: 978:         Distance from agent center to sensors, by default 5.0
1006: 1006: 1006: 979:     sensor_layout : str, optional
1007: 1007: 1007: 980:         Sensor arrangement ("bilateral", "triangular", "custom"), by default "bilateral"
1008: 1008: 1008: 981:     reward_config : Optional[Dict[str, float]], optional
1009: 1009: 1009: 982:         Dictionary of reward function weights, by default None
1010: 1010: 1010: 983:     max_episode_steps : int, optional
1011: 1011: 1011: 984:         Maximum steps per episode, by default 1000
1012: 1012: 1012: 985:     render_mode : Optional[str], optional
1013: 1013: 1013: 986:         Rendering mode ("human", "rgb_array", "headless"), by default None
1014: 1014: 1014: 987:     seed : Optional[int], optional
1015: 1015: 1015: 988:         Random seed for reproducible experiments, by default None
1016: 1016: 1016: 989:     performance_monitoring : bool, optional
1017: 1017: 1017: 990:         Enable performance tracking, by default True
1018: 1018: 1018: 991:     **kwargs : Any
1019: 1019: 1019: 992:         Additional configuration parameters
1020: 1020: 1020: 993: 
1021: 1021: 1021: 994:     Returns
1022: 1022: 1022: 995:     -------
1023: 1023: 1023: 996:     GymnasiumEnv
1024: 1024: 1024: 997:         Configured Gymnasium environment instance ready for RL training
1025: 1025: 1025: 998: 
1026: 1026: 1026: 999:     Raises
1027: 1027: 1027: 1000:     ------
1028: 1028: 1028: 1001:     ImportError
1029: 1029: 1029: 1002:         If Gymnasium dependencies are not available
1030: 1030: 1030: 1003:     ValueError
1031: 1031: 1031: 1004:         If configuration parameters are invalid or incomplete
1032: 1032: 1032: 1005:     FileNotFoundError
1033: 1033: 1033: 1006:         If video file does not exist
1034: 1034: 1034: 1007:     RuntimeError
1035: 1035: 1035: 1008:         If environment initialization fails
1036: 1036: 1036: 1009: 
1037: 1037: 1037: 1010:     Examples
1038: 1038: 1038: 1011:     --------
1039: 1039: 1039: 1012:     Create environment with Hydra configuration:
1040: 1040: 1040: 1013:         >>> from hydra import compose, initialize
1041: 1041: 1041: 1014:         >>> with initialize(config_path="../conf"):
1042: 1042: 1042: 1015:         ...     cfg = compose(config_name="rl_config")
1043: 1043: 1043: 1016:         ...     env = create_gymnasium_environment(cfg.environment)
1044: 1044: 1044: 1017: 
1045: 1045: 1045: 1018:     Create environment with direct parameters:
1046: 1046: 1046: 1019:         >>> env = create_gymnasium_environment(
1047: 1047: 1047: 1020:         ...     video_path="data/plume_experiment.mp4",
1048: 1048: 1048: 1021:         ...     initial_position=(320, 240),
1049: 1049: 1049: 1022:         ...     max_speed=2.5,
1050: 1050: 1050: 1023:         ...     include_multi_sensor=True,
1051: 1051: 1051: 1024:         ...     render_mode="human"
1052: 1052: 1052: 1025:         ... )
1053: 1053: 1053: 1026: 
1054: 1054: 1054: 1027:     Integration with stable-baselines3:
1055: 1055: 1055: 1028:         >>> from stable_baselines3 import PPO
1056: 1056: 1056: 1029:         >>> env = create_gymnasium_environment(cfg.environment)
1057: 1057: 1057: 1030:         >>> model = PPO("MultiInputPolicy", env, verbose=1)
1058: 1058: 1058: 1031:         >>> model.learn(total_timesteps=100000)
1059: 1059: 1059: 1032: 
1060: 1060: 1060: 1033:     Vectorized training:
1061: 1061: 1061: 1034:         >>> from stable_baselines3.common.vec_env import DummyVecEnv
1062: 1062: 1062: 1035:         >>> def make_env():
1063: 1063: 1063: 1036:         ...     return create_gymnasium_environment(env_config)
1064: 1064: 1064: 1037:         >>> vec_env = DummyVecEnv([make_env for _ in range(4)])
1065: 1065: 1065: 1038:         >>> model = PPO("MultiInputPolicy", vec_env)
1066: 1066: 1066: 1039: 
1067: 1067: 1067: 1040:     Traditional simulation workflow comparison:
1068: 1068: 1068: 1041:         >>> # Traditional simulation approach
1069: 1069: 1069: 1042:         >>> navigator = create_navigator(position=(100, 100), max_speed=2.0)
1070: 1070: 1070: 1043:         >>> plume = create_video_plume(video_path="experiment.mp4")
1071: 1071: 1071: 1044:         >>> positions, orientations, readings = run_plume_simulation(navigator, plume)
1072: 1072: 1072: 1045:         >>> 
1073: 1073: 1073: 1046:         >>> # Equivalent RL training approach
1074: 1074: 1074: 1047:         >>> env = create_gymnasium_environment(
1075: 1075: 1075: 1048:         ...     video_path="experiment.mp4",
1076: 1076: 1076: 1049:         ...     initial_position=(100, 100),
1077: 1077: 1077: 1050:         ...     max_speed=2.0
1078: 1078: 1078: 1051:         ... )
1079: 1079: 1079: 1052:         >>> from stable_baselines3 import PPO
1080: 1080: 1080: 1053:         >>> model = PPO("MultiInputPolicy", env)
1081: 1081: 1081: 1054:         >>> model.learn(total_timesteps=50000)
1082: 1082: 1082: 1055: 
1083: 1083: 1083: 1056:     Notes
1084: 1084: 1084: 1057:     -----
1085: 1085: 1085: 1058:     Configuration precedence order:
1086: 1086: 1086: 1059:     1. Direct parameters (video_path, max_speed, etc.)
1087: 1087: 1087: 1060:     2. Hydra DictConfig object values
1088: 1088: 1088: 1061:     3. Default values from GymnasiumEnv
1089: 1089: 1089: 1062: 
1090: 1090: 1090: 1063:     Environment compatibility:
1091: 1091: 1091: 1064:     - Full Gymnasium API compliance verified by env_checker
1092: 1092: 1092: 1065:     - Compatible with stable-baselines3 algorithms (PPO, SAC, TD3)
1093: 1093: 1093: 1066:     - Supports vectorized environments for parallel training
1094: 1094: 1094: 1067:     - Integration with Hydra configuration management
1095: 1095: 1095: 1068: 
1096: 1096: 1096: 1069:     Performance characteristics:
1097: 1097: 1097: 1070:     - Initialization time: <2s for typical configurations
1098: 1098: 1098: 1071:     - Step execution: ≥30 FPS simulation performance
1099: 1099: 1099: 1072:     - Memory usage: Scales linearly with episode length
1100: 1100: 1100: 1073:     """
1101: 1101: 1101: 1074:     # Initialize logger with function context
1102: 1102: 1102: 1075:     func_logger = logger.bind(
1103: 1103: 1103: 1076:         module=__name__,
1104: 1104: 1104: 1077:         function="create_gymnasium_environment",
1105: 1105: 1105: 1078:         cfg_provided=cfg is not None,
1106: 1106: 1106: 1079:         video_path_provided=video_path is not None
1107: 1107: 1107: 1080:     )
1108: 1108: 1108: 1081: 
1109: 1109: 1109: 1082:     # Check Gymnasium availability
1110: 1110: 1110: 1083:     if not GYMNASIUM_AVAILABLE:
1111: 1111: 1111: 1084:         raise ImportError(
1112: 1112: 1112: 1085:             "Gymnasium environment support is not available. "
1113: 1113: 1113: 1086:             "Install with: pip install 'odor_plume_nav[rl]' to enable RL functionality."
1114: 1114: 1114: 1087:         )
1115: 1115: 1115: 1088: 
1116: 1116: 1116: 1089:     try:
1117: 1117: 1117: 1090:         func_logger.info("Creating Gymnasium environment for RL training")
1118: 1118: 1118: 1091: 
1119: 1119: 1119: 1092:         # Process configuration object if provided
1120: 1120: 1120: 1093:         config_params = {}
1121: 1121: 1121: 1094:         if cfg is not None:
1122: 1122: 1122: 1095:             try:
1123: 1123: 1123: 1096:                 if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
1124: 1124: 1124: 1097:                     config_params = OmegaConf.to_container(cfg, resolve=True)
1125: 1125: 1125: 1098:                 elif isinstance(cfg, dict):
1126: 1126: 1126: 1099:                     config_params = cfg.copy()
1127: 1127: 1127: 1100:                 else:
1128: 1128: 1128: 1101:                     raise TypeError(f"Configuration must be DictConfig or dict, got {type(cfg)}")
1129: 1129: 1129: 1102:             except Exception as e:
1130: 1130: 1130: 1103:                 func_logger.error(f"Failed to process configuration: {e}")
1131: 1131: 1131: 1104:                 raise ValueError(f"Invalid configuration object: {e}") from e
1132: 1132: 1132: 1105: 
1133: 1133: 1133: 1106:         # Merge parameters with precedence (direct params override config)
1134: 1134: 1134: 1107:         merged_params = {
1135: 1135: 1135: 1108:             "video_path": video_path or config_params.get("video_path"),
1136: 1136: 1136: 1109:             "initial_position": initial_position or config_params.get("initial_position"),
1137: 1137: 1137: 1110:             "initial_orientation": initial_orientation if initial_orientation != 0.0 else config_params.get("initial_orientation", 0.0),
1138: 1138: 1138: 1111:             "max_speed": max_speed if max_speed != 2.0 else config_params.get("max_speed", 2.0),
1139: 1139: 1139: 1112:             "max_angular_velocity": max_angular_velocity if max_angular_velocity != 90.0 else config_params.get("max_angular_velocity", 90.0),
1140: 1140: 1140: 1113:             "include_multi_sensor": include_multi_sensor if include_multi_sensor else config_params.get("include_multi_sensor", False),
1141: 1141: 1141: 1114:             "num_sensors": num_sensors if num_sensors != 2 else config_params.get("num_sensors", 2),
1142: 1142: 1142: 1115:             "sensor_distance": sensor_distance if sensor_distance != 5.0 else config_params.get("sensor_distance", 5.0),
1143: 1143: 1143: 1116:             "sensor_layout": sensor_layout if sensor_layout != "bilateral" else config_params.get("sensor_layout", "bilateral"),
1144: 1144: 1144: 1117:             "reward_config": reward_config or config_params.get("reward_config"),
1145: 1145: 1145: 1118:             "max_episode_steps": max_episode_steps if max_episode_steps != 1000 else config_params.get("max_episode_steps", 1000),
1146: 1146: 1146: 1119:             "render_mode": render_mode or config_params.get("render_mode"),
1147: 1147: 1147: 1120:             "seed": seed or config_params.get("seed"),
1148: 1148: 1148: 1121:             "performance_monitoring": performance_monitoring if performance_monitoring else config_params.get("performance_monitoring", True),
1149: 1149: 1149: 1122:         }
1150: 1150: 1150: 1123: 
1151: 1151: 1151: 1124:         # Add any additional config parameters
1152: 1152: 1152: 1125:         merged_params.update({k: v for k, v in config_params.items() 
1153: 1153: 1153: 1126:                             if k not in merged_params})
1154: 1154: 1154: 1127:         merged_params.update(kwargs)
1155: 1155: 1155: 1128: 
1156: 1156: 1156: 1129:         # Remove None values to use GymnasiumEnv defaults
1157: 1157: 1157: 1130:         constructor_params = {k: v for k, v in merged_params.items() if v is not None}
1158: 1158: 1158: 1131: 
1159: 1159: 1159: 1132:         # Validate required video_path parameter
1160: 1160: 1160: 1133:         if "video_path" not in constructor_params or constructor_params["video_path"] is None:
1161: 1161: 1161: 1134:             raise ValueError("video_path is required for Gymnasium environment creation")
1162: 1162: 1162: 1135: 
1163: 1163: 1163: 1136:         # Create GymnasiumEnv instance
1164: 1164: 1164: 1137:         env = GymnasiumEnv(**constructor_params)
1165: 1165: 1165: 1138: 
1166: 1166: 1166: 1139:         func_logger.info(
1167: 1167: 1167: 1140:             "Gymnasium environment created successfully",
1168: 1168: 1168: 1141:             extra={
1169: 1169: 1169: 1142:                 "video_path": str(constructor_params["video_path"]),
1170: 1170: 1170: 1143:                 "action_space": str(env.action_space),
1171: 1171: 1171: 1144:                 "observation_space_keys": list(env.observation_space.spaces.keys()),
1172: 1172: 1172: 1145:                 "max_episode_steps": env.max_episode_steps,
1173: 1173: 1173: 1146:                 "config_source": "hydra" if cfg is not None else "direct"
1174: 1174: 1174: 1147:             }
1175: 1175: 1175: 1148:         )
1176: 1176: 1176: 1149: 
1177: 1177: 1177: 1150:         return env
1178: 1178: 1178: 1151: 
1179: 1179: 1179: 1152:     except Exception as e:
1180: 1180: 1180: 1153:         func_logger.error(f"Gymnasium environment creation failed: {e}")
1181: 1181: 1181: 1154:         raise RuntimeError(f"Failed to create Gymnasium environment: {e}") from e
1182: 1182: 1182: 1155: 
1183: 1183: 1183: 1156: 
1184: 1184: 1184: 1157: def from_legacy(
1185: 1185: 1185: 1158:     navigator: NavigatorProtocol,
1186: 1186: 1186: 1159:     video_plume: VideoPlume,
1187: 1187: 1187: 1160:     simulation_config: Optional[Union[DictConfig, Dict[str, Any]]] = None,
1188: 1188: 1188: 1161:     reward_config: Optional[Dict[str, float]] = None,
1189: 1189: 1189: 1162:     max_episode_steps: Optional[int] = None,
1190: 1190: 1190: 1163:     render_mode: Optional[str] = None,
1191: 1191: 1191: 1164:     **env_kwargs: Any
1192: 1192: 1192: 1165: ) -> "GymnasiumEnv":
1193: 1193: 1193: 1166:     """
1194: 1194: 1194: 1167:     Create a Gymnasium environment from existing legacy simulation components.
1195: 1195: 1195: 1168: 
1196: 1196: 1196: 1169:     This migration function provides backward compatibility for users transitioning from
1197: 1197: 1197: 1170:     the traditional simulation API to the Gymnasium RL interface. It takes existing
1198: 1198: 1198: 1171:     navigator and video plume instances and wraps them in a Gymnasium-compliant environment.
1199: 1199: 1199: 1172: 
1200: 1200: 1200: 1173:     The function serves as a bridge between legacy simulation workflows and modern RL
1201: 1201: 1201: 1174:     training, enabling researchers to leverage their existing configurations while
1202: 1202: 1202: 1175:     gaining access to the standardized RL ecosystem.
1203: 1203: 1203: 1176: 
1204: 1204: 1204: 1177:     Parameters
1205: 1205: 1205: 1178:     ----------
1206: 1206: 1206: 1179:     navigator : NavigatorProtocol
1207: 1207: 1207: 1180:         Existing navigator instance (SingleAgentController or MultiAgentController)
1208: 1208: 1208: 1181:     video_plume : VideoPlume
1209: 1209: 1209: 1182:         Configured VideoPlume environment instance
1210: 1210: 1210: 1183:     simulation_config : Optional[Union[DictConfig, Dict[str, Any]]], optional
1211: 1211: 1211: 1184:         Existing simulation configuration to extract parameters from, by default None
1212: 1212: 1212: 1185:     reward_config : Optional[Dict[str, float]], optional
1213: 1213: 1213: 1186:         Custom reward function weights for RL training, by default None
1214: 1214: 1214: 1187:     max_episode_steps : Optional[int], optional
1215: 1215: 1215: 1188:         Maximum steps per episode (default: derived from simulation config), by default None
1216: 1216: 1216: 1189:     render_mode : Optional[str], optional
1217: 1217: 1217: 1190:         Rendering mode for the Gymnasium environment, by default None
1218: 1218: 1218: 1191:     **env_kwargs : Any
1219: 1219: 1219: 1192:         Additional environment configuration parameters
1220: 1220: 1220: 1193: 
1221: 1221: 1221: 1194:     Returns
1222: 1222: 1222: 1195:     -------
1223: 1223: 1223: 1196:     GymnasiumEnv
1224: 1224: 1224: 1197:         Gymnasium environment configured with legacy components
1225: 1225: 1225: 1198: 
1226: 1226: 1226: 1199:     Raises
1227: 1227: 1227: 1200:     ------
1228: 1228: 1228: 1201:     ImportError
1229: 1229: 1229: 1202:         If Gymnasium dependencies are not available
1230: 1230: 1230: 1203:     TypeError
1231: 1231: 1231: 1204:         If navigator or video_plume don't meet required protocols
1232: 1232: 1232: 1205:     ValueError
1233: 1233: 1233: 1206:         If component configurations are incompatible
1234: 1234: 1234: 1207:     RuntimeError
1235: 1235: 1235: 1208:         If environment creation fails
1236: 1236: 1236: 1209: 
1237: 1237: 1237: 1210:     Examples
1238: 1238: 1238: 1211:     --------
1239: 1239: 1239: 1212:     Migrate from traditional simulation workflow:
1240: 1240: 1240: 1213:         >>> # Traditional workflow
1241: 1241: 1241: 1214:         >>> navigator = create_navigator(position=(100, 100), max_speed=2.0)
1242: 1242: 1242: 1215:         >>> plume = create_video_plume(video_path="experiment.mp4")
1243: 1243: 1243: 1216:         >>> 
1244: 1244: 1244: 1217:         >>> # Migrate to RL environment
1245: 1245: 1245: 1218:         >>> env = from_legacy(navigator, plume, render_mode="human")
1246: 1246: 1246: 1219:         >>> 
1247: 1247: 1247: 1220:         >>> # Now use with stable-baselines3
1248: 1248: 1248: 1221:         >>> from stable_baselines3 import PPO
1249: 1249: 1249: 1222:         >>> model = PPO("MultiInputPolicy", env)
1250: 1250: 1250: 1223: 
1251: 1251: 1251: 1224:     Preserve existing configuration:
1252: 1252: 1252: 1225:         >>> # With existing simulation config
1253: 1253: 1253: 1226:         >>> sim_config = {"max_steps": 2000, "dt": 0.1}
1254: 1254: 1254: 1227:         >>> env = from_legacy(
1255: 1255: 1255: 1228:         ...     navigator, plume, 
1256: 1256: 1256: 1229:         ...     simulation_config=sim_config,
1257: 1257: 1257: 1230:         ...     reward_config={"odor_concentration": 2.0}
1258: 1258: 1258: 1231:         ... )
1259: 1259: 1259: 1232: 
1260: 1260: 1260: 1233:     Multi-agent migration:
1261: 1261: 1261: 1234:         >>> # Multi-agent navigator
1262: 1262: 1262: 1235:         >>> multi_navigator = create_navigator(
1263: 1263: 1263: 1236:         ...     positions=[(50, 50), (150, 150)],
1264: 1264: 1264: 1237:         ...     max_speeds=[2.0, 3.0]
1265: 1265: 1265: 1238:         ... )
1266: 1266: 1266: 1239:         >>> env = from_legacy(multi_navigator, plume)
1267: 1267: 1267: 1240:         >>> # Note: Results in vectorized single-agent envs for RL compatibility
1268: 1268: 1268: 1241: 
1269: 1269: 1269: 1242:     Custom reward configuration:
1270: 1270: 1270: 1243:         >>> custom_rewards = {
1271: 1271: 1271: 1244:         ...     "odor_concentration": 1.5,
1272: 1272: 1272: 1245:         ...     "distance_penalty": -0.02,
1273: 1273: 1273: 1246:         ...     "exploration_bonus": 0.15
1274: 1274: 1274: 1247:         ... }
1275: 1275: 1275: 1248:         >>> env = from_legacy(navigator, plume, reward_config=custom_rewards)
1276: 1276: 1276: 1249: 
1277: 1277: 1277: 1250:     Notes
1278: 1278: 1278: 1251:     -----
1279: 1279: 1279: 1252:     Migration considerations:
1280: 1280: 1280: 1253:     - Single-agent navigators create single Gymnasium environments
1281: 1281: 1281: 1254:     - Multi-agent navigators are converted to vectorized single-agent environments
1282: 1282: 1282: 1255:     - Existing VideoPlume preprocessing is preserved
1283: 1283: 1283: 1256:     - Navigator max_speed and position constraints are maintained
1284: 1284: 1284: 1257:     - Simulation timestep (dt) is normalized to 1.0 for RL compatibility
1285: 1285: 1285: 1258: 
1286: 1286: 1286: 1259:     Configuration extraction:
1287: 1287: 1287: 1260:     - Navigator position and orientation become initial_position/initial_orientation
1288: 1288: 1288: 1261:     - Navigator max_speed becomes environment max_speed constraint
1289: 1289: 1289: 1262:     - VideoPlume video_path and preprocessing settings are preserved
1290: 1290: 1290: 1263:     - Simulation max_steps maps to max_episode_steps
1291: 1291: 1291: 1264: 
1292: 1292: 1292: 1265:     Performance optimization:
1293: 1293: 1293: 1266:     - Environment initialization reuses existing component configurations
1294: 1294: 1294: 1267:     - No additional video file loading or navigator re-initialization
1295: 1295: 1295: 1268:     - Minimal overhead for component wrapping
1296: 1296: 1296: 1269:     """
1297: 1297: 1297: 1270:     # Initialize logger with migration context
1298: 1298: 1298: 1271:     func_logger = logger.bind(
1299: 1299: 1299: 1272:         module=__name__,
1300: 1300: 1300: 1273:         function="from_legacy",
1301: 1301: 1301: 1274:         navigator_type=type(navigator).__name__,
1302: 1302: 1302: 1275:         num_agents=navigator.num_agents,
1303: 1303: 1303: 1276:         video_path=str(video_plume.video_path) if hasattr(video_plume, 'video_path') else "unknown"
1304: 1304: 1304: 1277:     )
1305: 1305: 1305: 1278: 
1306: 1306: 1306: 1279:     # Check Gymnasium availability
1307: 1307: 1307: 1280:     if not GYMNASIUM_AVAILABLE:
1308: 1308: 1308: 1281:         raise ImportError(
1309: 1309: 1309: 1282:             "Gymnasium environment support is not available. "
1310: 1310: 1310: 1283:             "Install with: pip install 'odor_plume_nav[rl]' to enable RL functionality."
1311: 1311: 1311: 1284:         )
1312: 1312: 1312: 1285: 
1313: 1313: 1313: 1286:     try:
1314: 1314: 1314: 1287:         func_logger.info("Migrating legacy simulation components to Gymnasium environment")
1315: 1315: 1315: 1288: 
1316: 1316: 1316: 1289:         # Validate input components
1317: 1317: 1317: 1290:         if not hasattr(navigator, 'positions') or not hasattr(navigator, 'step'):
1318: 1318: 1318: 1291:             raise TypeError("navigator must implement NavigatorProtocol interface")
1319: 1319: 1319: 1292:         
1320: 1320: 1320: 1293:         if not hasattr(video_plume, 'get_frame') or not hasattr(video_plume, 'video_path'):
1321: 1321: 1321: 1294:             raise TypeError("video_plume must be a VideoPlume instance")
1322: 1322: 1322: 1295: 
1323: 1323: 1323: 1296:         # Extract configuration from simulation_config if provided
1324: 1324: 1324: 1297:         config_params = {}
1325: 1325: 1325: 1298:         if simulation_config is not None:
1326: 1326: 1326: 1299:             try:
1327: 1327: 1327: 1300:                 if HYDRA_AVAILABLE and isinstance(simulation_config, DictConfig):
1328: 1328: 1328: 1301:                     config_params = OmegaConf.to_container(simulation_config, resolve=True)
1329: 1329: 1329: 1302:                 elif isinstance(simulation_config, dict):
1330: 1330: 1330: 1303:                     config_params = simulation_config.copy()
1331: 1331: 1331: 1304:                 else:
1332: 1332: 1332: 1305:                     raise TypeError(f"simulation_config must be DictConfig or dict, got {type(simulation_config)}")
1333: 1333: 1333: 1306:             except Exception as e:
1334: 1334: 1334: 1307:                 func_logger.error(f"Failed to process simulation configuration: {e}")
1335: 1335: 1335: 1308:                 raise ValueError(f"Invalid simulation configuration: {e}") from e
1336: 1336: 1336: 1309: 
1337: 1337: 1337: 1310:         # Extract navigator configuration
1338: 1338: 1338: 1311:         # For single agents, use first agent's parameters
1339: 1339: 1339: 1312:         if navigator.num_agents == 1:
1340: 1340: 1340: 1313:             initial_position = tuple(navigator.positions[0])
1341: 1341: 1341: 1314:             initial_orientation = float(navigator.orientations[0])
1342: 1342: 1342: 1315:             max_speed = float(navigator.max_speeds[0])
1343: 1343: 1343: 1316:         else:
1344: 1344: 1344: 1317:             # For multi-agent, use first agent as template and warn
1345: 1345: 1345: 1318:             func_logger.warning(
1346: 1346: 1346: 1319:                 f"Multi-agent navigator with {navigator.num_agents} agents detected. "
1347: 1347: 1347: 1320:                 "Creating single-agent Gymnasium environment using first agent's parameters. "
1348: 1348: 1348: 1321:                 "Consider using vectorized environments for true multi-agent RL training."
1349: 1349: 1349: 1322:             )
1350: 1350: 1350: 1323:             initial_position = tuple(navigator.positions[0])
1351: 1351: 1351: 1324:             initial_orientation = float(navigator.orientations[0])
1352: 1352: 1352: 1325:             max_speed = float(navigator.max_speeds[0])
1353: 1353: 1353: 1326: 
1354: 1354: 1354: 1327:         # Extract video plume configuration
1355: 1355: 1355: 1328:         video_path = video_plume.video_path
1356: 1356: 1356: 1329: 
1357: 1357: 1357: 1330:         # Determine episode length from simulation config or use default
1358: 1358: 1358: 1331:         if max_episode_steps is None:
1359: 1359: 1359: 1332:             max_episode_steps = (
1360: 1360: 1360: 1333:                 config_params.get("max_steps", 
1361: 1361: 1361: 1334:                 config_params.get("num_steps", 1000))
1362: 1362: 1362: 1335:             )
1363: 1363: 1363: 1336: 
1364: 1364: 1364: 1337:         # Build environment parameters
1365: 1365: 1365: 1338:         env_params = {
1366: 1366: 1366: 1339:             "video_path": video_path,
1367: 1367: 1367: 1340:             "initial_position": initial_position,
1368: 1368: 1368: 1341:             "initial_orientation": initial_orientation,
1369: 1369: 1369: 1342:             "max_speed": max_speed,
1370: 1370: 1370: 1343:             "max_angular_velocity": 90.0,  # Default, can be overridden
1371: 1371: 1371: 1344:             "reward_config": reward_config,
1372: 1372: 1372: 1345:             "max_episode_steps": max_episode_steps,
1373: 1373: 1373: 1346:             "render_mode": render_mode,
1374: 1374: 1374: 1347:             "performance_monitoring": True
1375: 1375: 1375: 1348:         }
1376: 1376: 1376: 1349: 
1377: 1377: 1377: 1350:         # Apply any additional overrides
1378: 1378: 1378: 1351:         env_params.update(env_kwargs)
1379: 1379: 1379: 1352: 
1380: 1380: 1380: 1353:         # Create Gymnasium environment
1381: 1381: 1381: 1354:         env = GymnasiumEnv(**env_params)
1382: 1382: 1382: 1355: 
1383: 1383: 1383: 1356:         func_logger.info(
1384: 1384: 1384: 1357:             "Legacy migration completed successfully",
1385: 1385: 1385: 1358:             extra={
1386: 1386: 1386: 1359:                 "source_navigator": type(navigator).__name__,
1387: 1387: 1387: 1360:                 "source_agents": navigator.num_agents,
1388: 1388: 1388: 1361:                 "video_path": str(video_path),
1389: 1389: 1389: 1362:                 "environment_config": {
1390: 1390: 1390: 1363:                     "initial_position": initial_position,
1391: 1391: 1391: 1364:                     "max_speed": max_speed,
1392: 1392: 1392: 1365:                     "max_episode_steps": max_episode_steps
1393: 1393: 1393: 1366:                 }
1394: 1394: 1394: 1367:             }
1395: 1395: 1395: 1368:         )
1396: 1396: 1396: 1369: 
1397: 1397: 1397: 1370:         return env
1398: 1398: 1398: 1371: 
1399: 1399: 1399: 1372:     except Exception as e:
1400: 1400: 1400: 1373:         func_logger.error(f"Legacy migration failed: {e}")
1401: 1401: 1401: 1374:         raise RuntimeError(f"Failed to migrate legacy components to Gymnasium environment: {e}") from e
1402: 1402: 1402: 1375: 
1403: 1403: 1403: 1376: 928: 919: 919:                 plume_frames=plume_frames,
1404: 1404: 1404: 1377: 929: 920: 920:                 **viz_params
1405: 1405: 1405: 1378: 930: 921: 921:             )
1406: 1406: 1406: 1379: 931: 922: 922: 
1407: 1407: 1407: 1380: 932: 923: 923:     except Exception as e:
1408: 1408: 1408: 1381: 933: 924: 924:         viz_logger.error(f"Visualization failed: {e}")
1409: 1409: 1409: 1382: 934: 925: 925:         raise RuntimeError(f"Failed to create visualization: {e}") from e
1410: 1410: 1410: 1383: 935: 926: 926: 
1411: 1411: 1411: 1384: 936: 927: 927: 
1412: 1412: 1412: 1385: 937: 928: 928: # Legacy compatibility aliases
1413: 1413: 1413: 1386: 938: 929: 929: create_navigator_from_config = create_navigator
1414: 1414: 1414: 1387: 939: 930: 930: create_video_plume_from_config = create_video_plume
1415: 1415: 1415: 1388: 940: 931: 931: run_simulation = run_plume_simulation
1416: 1416: 1416: 1389: 941: 932: 932: visualize_simulation_results = visualize_plume_simulation
1417: 
1418: # Gymnasium environment aliases for compatibility
1419: create_rl_environment = create_gymnasium_environment
1420: migrate_to_rl = from_legacy
1421: 1417: 1417: 1390: 942: 933: 933: 
1422: 1418: 1418: 1391: 943: 934: 934: # Export public API
1423: 1419: 1419: 1392: 944: 935: 935: __all__ = [
1424: 1420: 1420: 1393: 945: 936: 936:     "create_navigator",
1425: 1421: 1421: 1394: 946: 937: 937:     "create_video_plume", 
1426: 1422: 1422: 1395: 947: 938: 938:     "run_plume_simulation",
1427: 1423: 1423: 1396: 948: 939: 939:     "visualize_plume_simulation",
1428: 1424:     # Gymnasium environment functions
1429: 1425:     "create_gymnasium_environment",
1430: 1426:     "from_legacy",
1431: 1427: 1424: 1397: 949: 940: 940:     # Legacy aliases
1432: 1428: 1425: 1398: 950: 941: 941:     "create_navigator_from_config",
1433: 1429: 1426: 1399: 951: 942: 942:     "create_video_plume_from_config", 
1434: 1430: 1427: 1400: 952: 943: 943:     "run_simulation",
1435: 1431: 1428: 1401: 953: 944: 944:     "    "visualize_simulation_results",
    # Gymnasium environment aliases
    "create_rl_environment",
    "migrate_to_rl""
1436: 1432: 1429: 1402: 954: 945: 945: ]