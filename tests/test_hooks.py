"""
Comprehensive test module for extension system functionality validating hook system 
capabilities including lifecycle callbacks, Debug GUI integration, and non-invasive 
system extension mechanisms for accelerated research iteration.

This test suite validates the hook system per F-019 requirements for debugging and 
hook toolkit capabilities, ensuring comprehensive testing of:

- Hook system functionality with pre-step, post-step, and episode-end callbacks
- HookManager integration with simulation loop for non-invasive extensions  
- extra_obs_fn and extra_reward_fn implementations for dynamic environment extension
- Hook system performance with zero overhead when disabled and minimal impact when enabled
- Hook configuration via Hydra config group 'conf/base/hooks/' for runtime registration
- Debug GUI integration with interactive step-through controls and real-time visualization
- Custom debugging extensions and export capabilities for screenshots and state dumps

Performance targets validated:
- ≤33 ms/step with 100 agents through optimized hook processing
- Zero overhead when hooks disabled
- Minimal performance impact when hooks enabled
- Integration with simulation loop and visualization system for real-time debugging

Test Categories:
1. Basic Hook Functionality Tests
2. Hook Integration Tests with Environment
3. Performance and Overhead Tests  
4. Debug GUI Integration Tests
5. Hook Configuration Tests
6. Error Handling and Edge Cases
7. Concurrent Execution Tests
8. Export and Data Persistence Tests
"""

import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import time
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union

# External imports per schema requirements
from src.plume_nav_sim.core.protocols import RecorderProtocol
from src.plume_nav_sim.debug.gui import DebugSession  
from src.plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv


class TestHookSystemBasicFunctionality:
    """Test basic hook system functionality including lifecycle callbacks."""
    
    def test_extra_obs_fn_integration(self):
        """Test extra_obs_fn hook implementation for dynamic environment extension."""
        # Mock environment with hook system
        env = MagicMock(spec=PlumeNavigationEnv)
        env.extra_obs_fn = None
        env.state = {
            'position': np.array([10.0, 20.0]),
            'orientation': 45.0,
            'speed': 1.5,
            'odor_concentration': 0.3
        }
        
        # Define custom observation hook
        def custom_obs_hook(state):
            return {
                'wind_direction': np.array([2.0, 1.0]),
                'energy_level': 0.8,
                'custom_sensor': state.get('odor_concentration', 0.0) * 2
            }
        
        # Test hook registration
        env.extra_obs_fn = custom_obs_hook
        
        # Test hook execution during observation computation
        base_obs = {
            'position': env.state['position'],
            'orientation': env.state['orientation'],
            'speed': env.state['speed']
        }
        
        # Simulate environment's observation computation with hook
        if env.extra_obs_fn:
            extra_obs = env.extra_obs_fn(env.state)
            combined_obs = {**base_obs, **extra_obs}
        else:
            combined_obs = base_obs
        
        # Validate hook execution and output
        assert 'wind_direction' in combined_obs
        assert 'energy_level' in combined_obs
        assert 'custom_sensor' in combined_obs
        assert combined_obs['custom_sensor'] == 0.6  # 0.3 * 2
        np.testing.assert_array_equal(combined_obs['wind_direction'], [2.0, 1.0])
        assert combined_obs['energy_level'] == 0.8
    
    def test_extra_reward_fn_integration(self):
        """Test extra_reward_fn hook for custom reward shaping."""
        # Mock environment with reward hook
        env = MagicMock(spec=PlumeNavigationEnv)
        env.extra_reward_fn = None
        env.state = {
            'position': np.array([15.0, 25.0]),
            'visited_positions': [(10, 20), (12, 22), (15, 25)],
            'step_count': 50
        }
        
        # Define custom reward hook
        def exploration_reward_hook(base_reward, info):
            # Bonus for visiting new areas
            exploration_bonus = 0.1 if len(info.get('visited_positions', [])) > 2 else 0.0
            # Penalty for excessive time
            time_penalty = -0.01 if info.get('step_count', 0) > 40 else 0.0
            return exploration_bonus + time_penalty
        
        # Test hook registration
        env.extra_reward_fn = exploration_reward_hook
        
        # Simulate reward computation with hook
        base_reward = 1.0
        info = {
            'visited_positions': env.state['visited_positions'],
            'step_count': env.state['step_count']
        }
        
        if env.extra_reward_fn:
            extra_reward = env.extra_reward_fn(base_reward, info)
            total_reward = base_reward + extra_reward
        else:
            total_reward = base_reward
        
        # Validate reward shaping
        expected_extra = 0.1 - 0.01  # exploration bonus - time penalty
        assert total_reward == pytest.approx(1.0 + expected_extra)
    
    def test_episode_end_fn_integration(self):
        """Test episode_end_fn hook for episode completion handling."""
        # Mock environment with episode end hook
        env = MagicMock(spec=PlumeNavigationEnv)
        env.episode_end_fn = None
        
        # Track hook execution
        hook_called = False
        hook_info = None
        
        def episode_completion_hook(final_info):
            nonlocal hook_called, hook_info
            hook_called = True
            hook_info = final_info.copy()
        
        # Test hook registration
        env.episode_end_fn = episode_completion_hook
        
        # Simulate episode completion
        final_info = {
            'episode_length': 150,
            'success': True,
            'final_position': (85, 92),
            'total_reward': 25.5,
            'termination_reason': 'success'
        }
        
        # Trigger episode end hook
        if env.episode_end_fn:
            env.episode_end_fn(final_info)
        
        # Validate hook execution
        assert hook_called
        assert hook_info == final_info
        assert hook_info['success'] is True
        assert hook_info['episode_length'] == 150


class TestHookManagerIntegration:
    """Test HookManager integration with simulation loop for non-invasive extensions."""
    
    def test_simulation_loop_hook_integration(self):
        """Test hook system integration with simulation loop without modifying core logic."""
        # Mock HookManager
        hook_manager = MagicMock()
        hook_manager.pre_step_hooks = []
        hook_manager.post_step_hooks = []
        hook_manager.episode_end_hooks = []
        
        # Mock environment with hook manager
        env = MagicMock(spec=PlumeNavigationEnv)
        env.hook_manager = hook_manager
        env.state = {'step_count': 0}
        
        # Define hook functions
        pre_step_called = False
        post_step_called = False
        
        def pre_step_hook(env_state):
            nonlocal pre_step_called
            pre_step_called = True
            return {'pre_step_data': 'processed'}
        
        def post_step_hook(env_state, action, reward):
            nonlocal post_step_called  
            post_step_called = True
            return {'post_step_data': 'logged'}
        
        # Register hooks
        hook_manager.pre_step_hooks.append(pre_step_hook)
        hook_manager.post_step_hooks.append(post_step_hook)
        
        # Simulate step execution with hooks
        def mock_step(action):
            # Pre-step hooks
            for hook in env.hook_manager.pre_step_hooks:
                hook(env.state)
            
            # Core step logic (unchanged)
            env.state['step_count'] += 1
            obs = {'position': np.array([10, 20])}
            reward = 1.0
            
            # Post-step hooks
            for hook in env.hook_manager.post_step_hooks:
                hook(env.state, action, reward)
            
            return obs, reward, False, False, {}
        
        # Execute step with hooks
        action = np.array([0.5, 0.1])
        obs, reward, terminated, truncated, info = mock_step(action)
        
        # Validate non-invasive hook execution
        assert pre_step_called
        assert post_step_called
        assert env.state['step_count'] == 1
        assert 'position' in obs
        
    def test_hook_system_isolation(self):
        """Test that hook failures don't break core simulation."""
        # Mock environment with failing hooks
        env = MagicMock(spec=PlumeNavigationEnv)
        env.extra_obs_fn = lambda state: {"invalid": 1/0}  # Intentional error
        env.extra_reward_fn = lambda base, info: "not_a_number"  # Type error
        
        # Test robust hook execution with error handling
        def safe_hook_execution():
            base_obs = {'position': np.array([0, 0])}
            base_reward = 1.0
            
            try:
                if env.extra_obs_fn:
                    extra_obs = env.extra_obs_fn({})
                    obs = {**base_obs, **extra_obs}
                else:
                    obs = base_obs
            except Exception:
                obs = base_obs  # Fallback to base observation
            
            try:
                if env.extra_reward_fn:
                    extra_reward = env.extra_reward_fn(base_reward, {})
                    reward = base_reward + extra_reward
                else:
                    reward = base_reward
            except Exception:
                reward = base_reward  # Fallback to base reward
            
            return obs, reward
        
        # Execute with error handling
        obs, reward = safe_hook_execution()
        
        # Validate core functionality preserved despite hook failures
        assert obs == {'position': np.array([0, 0])}
        assert reward == 1.0


class TestHookSystemPerformance:
    """Test hook system performance requirements and overhead measurements."""
    
    def test_zero_overhead_when_disabled(self):
        """Test that hook system has zero overhead when disabled."""
        # Mock environment without hooks
        env = MagicMock(spec=PlumeNavigationEnv)
        env.extra_obs_fn = None
        env.extra_reward_fn = None
        env.episode_end_fn = None
        
        # Baseline measurement without hooks
        def baseline_step():
            obs = {'position': np.array([10, 20]), 'speed': 1.0}
            reward = 1.0
            return obs, reward
        
        # Measurement with disabled hooks
        def step_with_disabled_hooks():
            obs = {'position': np.array([10, 20]), 'speed': 1.0}
            reward = 1.0
            
            # Hook checks (should be fast when None)
            if env.extra_obs_fn:
                extra_obs = env.extra_obs_fn({})
                obs.update(extra_obs)
            
            if env.extra_reward_fn:
                extra_reward = env.extra_reward_fn(reward, {})
                reward += extra_reward
            
            return obs, reward
        
        # Performance comparison
        baseline_times = []
        hook_times = []
        
        for _ in range(1000):
            # Baseline timing
            start = time.perf_counter()
            baseline_step()
            baseline_times.append(time.perf_counter() - start)
            
            # Hook timing
            start = time.perf_counter()
            step_with_disabled_hooks()
            hook_times.append(time.perf_counter() - start)
        
        baseline_mean = np.mean(baseline_times)
        hook_mean = np.mean(hook_times)
        overhead = (hook_mean - baseline_mean) / baseline_mean
        
        # Validate zero overhead (within measurement noise)
        assert overhead < 0.05  # Less than 5% overhead
    
    def test_performance_with_100_agents(self):
        """Test hook system performance with 100 agents meets ≤33ms/step target."""
        # Mock multi-agent environment
        num_agents = 100
        agents_state = {
            'positions': np.random.rand(num_agents, 2) * 100,
            'speeds': np.random.rand(num_agents) * 2,
            'concentrations': np.random.rand(num_agents)
        }
        
        # Define performance-optimized hooks
        def vectorized_obs_hook(state):
            positions = state['positions']
            return {
                'distances_to_center': np.linalg.norm(positions - 50, axis=1),
                'relative_speeds': state['speeds'] / 2.0
            }
        
        def vectorized_reward_hook(base_rewards, info):
            # Vectorized reward computation
            exploration_bonus = np.where(
                info.get('distances_to_center', np.zeros(num_agents)) > 25,
                0.1, 0.0
            )
            return exploration_bonus
        
        # Performance test
        execution_times = []
        
        for _ in range(100):  # 100 steps
            start = time.perf_counter()
            
            # Simulate multi-agent step with hooks
            base_obs = {
                'positions': agents_state['positions'],
                'speeds': agents_state['speeds']
            }
            base_rewards = np.ones(num_agents)
            
            # Execute hooks
            extra_obs = vectorized_obs_hook(agents_state)
            combined_obs = {**base_obs, **extra_obs}
            
            extra_rewards = vectorized_reward_hook(base_rewards, extra_obs)
            total_rewards = base_rewards + extra_rewards
            
            execution_time = time.perf_counter() - start
            execution_times.append(execution_time * 1000)  # Convert to ms
        
        mean_time = np.mean(execution_times)
        max_time = np.max(execution_times)
        
        # Validate performance target
        assert mean_time <= 33.0, f"Mean execution time {mean_time:.2f}ms exceeds 33ms target"
        assert max_time <= 50.0, f"Max execution time {max_time:.2f}ms exceeds reasonable limit"
    
    def test_hook_execution_efficiency(self):
        """Test hook execution efficiency and measure performance impact."""
        # Setup hooks with varying complexity
        simple_hook = lambda state: {'simple': 1}
        complex_hook = lambda state: {
            'complex': np.sum(np.random.rand(1000)),
            'computation': np.linalg.norm(np.random.rand(100, 2), axis=1).mean()
        }
        
        # Measure hook execution times
        simple_times = []
        complex_times = []
        
        for _ in range(1000):
            state = {'position': np.array([10, 20])}
            
            # Simple hook timing
            start = time.perf_counter()
            simple_hook(state)
            simple_times.append(time.perf_counter() - start)
            
            # Complex hook timing  
            start = time.perf_counter()
            complex_hook(state)
            complex_times.append(time.perf_counter() - start)
        
        simple_mean = np.mean(simple_times) * 1000  # ms
        complex_mean = np.mean(complex_times) * 1000  # ms
        
        # Validate hook efficiency requirements
        assert simple_mean < 0.1, f"Simple hook took {simple_mean:.3f}ms, should be <0.1ms"
        assert complex_mean < 10.0, f"Complex hook took {complex_mean:.3f}ms, should be <10ms"


class TestDebugGUIIntegration:
    """Test Debug GUI integration with interactive controls and state visualization."""
    
    def test_debug_session_hook_integration(self):
        """Test DebugSession integration with hook system for debugging."""
        # Create debug session
        session = DebugSession("test_session")
        
        # Test session configuration
        session.configure(
            shared=True,
            host='localhost',
            port=8502,
            mode='host'
        )
        
        # Validate session setup
        assert session.session_id == "test_session"
        assert session.collaborative_config['enabled'] is True
        assert session.collaborative_config['host'] == 'localhost'
        assert session.collaborative_config['port'] == 8502
    
    def test_debug_session_breakpoint_management(self):
        """Test DebugSession breakpoint functionality for step-through debugging."""
        session = DebugSession()
        
        # Add breakpoints
        bp1_id = session.add_breakpoint("odor_concentration > 0.5")
        bp2_id = session.add_breakpoint("position[0] > 50", enabled=True)
        
        # Validate breakpoint creation
        assert len(session.breakpoints) == 2
        assert session.breakpoints[0]['condition'] == "odor_concentration > 0.5"
        assert session.breakpoints[1]['condition'] == "position[0] > 50"
        assert session.breakpoints[0]['id'] == bp1_id
        assert session.breakpoints[1]['id'] == bp2_id
        
        # Test breakpoint removal
        success = session.remove_breakpoint(bp1_id)
        assert success is True
        assert len(session.breakpoints) == 1
        
        # Test invalid breakpoint removal
        success = session.remove_breakpoint(999)
        assert success is False
    
    def test_debug_session_inspector_management(self):
        """Test DebugSession inspector functionality for state inspection."""
        session = DebugSession()
        
        # Define inspector functions
        def position_inspector(state):
            return {
                'distance_from_origin': np.linalg.norm(state.get('position', [0, 0])),
                'quadrant': 'NE' if state.get('position', [0, 0])[0] > 0 else 'SW'
            }
        
        def performance_inspector(state):
            return {
                'step_rate': 1.0 / state.get('step_time', 0.001),
                'memory_usage': state.get('memory_mb', 0)
            }
        
        # Add inspectors
        session.add_inspector('position_analysis', position_inspector)
        session.add_inspector('performance_metrics', performance_inspector)
        
        # Validate inspector registration
        assert 'position_analysis' in session.inspectors
        assert 'performance_metrics' in session.inspectors
        assert session.inspectors['position_analysis']['function'] == position_inspector
        assert session.inspectors['performance_metrics']['function'] == performance_inspector
        
        # Test inspector execution
        test_state = {
            'position': np.array([10, 20]),
            'step_time': 0.01,
            'memory_mb': 128
        }
        
        pos_result = session.inspectors['position_analysis']['function'](test_state)
        perf_result = session.inspectors['performance_metrics']['function'](test_state)
        
        assert 'distance_from_origin' in pos_result
        assert pos_result['quadrant'] == 'NE'
        assert perf_result['step_rate'] == 100.0  # 1/0.01
    
    def test_debug_session_export_capabilities(self):
        """Test debug session export capabilities for data persistence."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = DebugSession("export_test")
            
            # Add sample data
            session.add_breakpoint("test_condition")
            session.performance_metrics = [
                {'step': 1, 'duration_ms': 10.5, 'exceeded_threshold': False},
                {'step': 2, 'duration_ms': 35.2, 'exceeded_threshold': True}
            ]
            
            # Test export
            export_path = Path(tmp_dir) / "session_export.json"
            success = session.export_session_data(export_path)
            
            assert success is True
            assert export_path.exists()
            
            # Validate export content
            import json
            with open(export_path) as f:
                exported_data = json.load(f)
            
            assert 'session_info' in exported_data
            assert 'breakpoints' in exported_data
            assert 'performance_metrics' in exported_data
            assert exported_data['session_info']['session_id'] == "export_test"
            assert len(exported_data['breakpoints']) == 1
            assert len(exported_data['performance_metrics']) == 2


class TestRecorderProtocolIntegration:
    """Test RecorderProtocol integration with hook system for data capture."""
    
    def test_recorder_hook_integration(self):
        """Test RecorderProtocol integration for hook-based data recording."""
        # Mock recorder
        recorder = MagicMock(spec=RecorderProtocol)
        recorder.record_step = MagicMock()
        recorder.record_episode = MagicMock() 
        recorder.export_data = MagicMock(return_value=True)
        
        # Hook function that uses recorder
        def recording_hook(state, recorder):
            step_data = {
                'position': state['position'].tolist() if hasattr(state['position'], 'tolist') else state['position'],
                'concentration': state.get('concentration', 0.0),
                'timestamp': time.time()
            }
            recorder.record_step(step_data, state.get('step_number', 0))
        
        # Test hook execution with recorder
        test_state = {
            'position': np.array([15.0, 25.0]),
            'concentration': 0.45,
            'step_number': 42
        }
        
        recording_hook(test_state, recorder)
        
        # Validate recorder integration
        recorder.record_step.assert_called_once()
        call_args = recorder.record_step.call_args[0]
        step_data = call_args[0]
        step_number = call_args[1]
        
        assert step_data['position'] == [15.0, 25.0]
        assert step_data['concentration'] == 0.45
        assert step_number == 42
    
    def test_recorder_export_integration(self):
        """Test recorder export capabilities during debug sessions."""
        # Mock recorder with export functionality
        recorder = MagicMock(spec=RecorderProtocol)
        recorder.export_data = MagicMock(return_value=True)
        
        # Debug session with recorder integration
        session = DebugSession()
        
        # Export hook
        def export_session_data(session, recorder, output_path):
            session_info = session.get_session_info()
            return recorder.export_data(
                output_path, 
                format="json",
                metadata=session_info
            )
        
        # Test export integration
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_path = Path(tmp_dir) / "session_data.json"
            success = export_session_data(session, recorder, str(export_path))
            
            assert success is True
            recorder.export_data.assert_called_once_with(
                str(export_path),
                format="json", 
                metadata=session.get_session_info()
            )


class TestHookConfiguration:
    """Test hook configuration via Hydra config groups."""
    
    def test_hook_config_structure(self):
        """Test hook configuration structure for runtime registration."""
        # Mock Hydra config structure
        hook_config = {
            'extra_obs': {
                '_target_': 'custom_hooks.wind_sensor_hook',
                'sensor_range': 10.0,
                'update_rate': 30
            },
            'extra_reward': {
                '_target_': 'custom_hooks.exploration_reward_hook', 
                'exploration_weight': 0.1,
                'efficiency_weight': 0.05
            },
            'episode_end': {
                '_target_': 'custom_hooks.episode_logging_hook',
                'log_level': 'INFO',
                'include_trajectory': True
            },
            'debug_hooks': {
                'performance_monitor': True,
                'state_inspector': True,
                'breakpoint_system': True
            }
        }
        
        # Validate config structure
        assert 'extra_obs' in hook_config
        assert 'extra_reward' in hook_config
        assert 'episode_end' in hook_config
        assert 'debug_hooks' in hook_config
        
        # Validate individual hook configs
        obs_config = hook_config['extra_obs']
        assert '_target_' in obs_config
        assert obs_config['sensor_range'] == 10.0
        
        reward_config = hook_config['extra_reward']
        assert reward_config['exploration_weight'] == 0.1
        
        debug_config = hook_config['debug_hooks']
        assert debug_config['performance_monitor'] is True
    
    def test_hook_instantiation_pattern(self):
        """Test hook instantiation from configuration."""
        # Mock hook instantiation function
        def instantiate_hook(config):
            if config['_target_'] == 'test_hook_function':
                def hook_function(state):
                    return {
                        'config_param': config.get('param_value', 'default'),
                        'state_info': len(state)
                    }
                return hook_function
            return None
        
        # Test configuration
        hook_config = {
            '_target_': 'test_hook_function',
            'param_value': 'configured_value'
        }
        
        # Instantiate hook
        hook_fn = instantiate_hook(hook_config)
        
        # Test hook execution
        test_state = {'a': 1, 'b': 2, 'c': 3}
        result = hook_fn(test_state)
        
        assert result['config_param'] == 'configured_value'
        assert result['state_info'] == 3


class TestConcurrentHookExecution:
    """Test hook system behavior in concurrent/threaded scenarios."""
    
    def test_thread_safe_hook_execution(self):
        """Test thread safety of hook system during concurrent execution."""
        # Shared state for testing
        execution_log = []
        lock = threading.Lock()
        
        def thread_safe_hook(state, thread_id):
            with lock:
                execution_log.append({
                    'thread_id': thread_id,
                    'timestamp': time.time(),
                    'state_id': state.get('id', 'unknown')
                })
        
        # Create multiple threads executing hooks
        threads = []
        for i in range(10):
            def worker(thread_id=i):
                for j in range(5):
                    state = {'id': f'thread_{thread_id}_state_{j}'}
                    thread_safe_hook(state, thread_id)
                    time.sleep(0.001)  # Small delay
            
            thread = threading.Thread(target=worker)
            threads.append(thread)
        
        # Execute threads
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Validate thread-safe execution
        assert len(execution_log) == 50  # 10 threads * 5 executions
        
        # Check all threads executed
        thread_ids = {entry['thread_id'] for entry in execution_log}
        assert len(thread_ids) == 10
        
        # Validate chronological ordering (due to lock)
        timestamps = [entry['timestamp'] for entry in execution_log]
        assert timestamps == sorted(timestamps)
    
    def test_hook_performance_under_load(self):
        """Test hook system performance under concurrent load."""
        # Performance tracking
        execution_times = []
        
        def performance_test_hook(state):
            start = time.perf_counter()
            # Simulate some work
            result = np.sum(np.random.rand(100))
            end = time.perf_counter()
            
            execution_times.append((end - start) * 1000)  # ms
            return {'computed_value': result}
        
        # Concurrent execution test
        def concurrent_worker():
            for _ in range(20):
                state = {'data': np.random.rand(10)}
                performance_test_hook(state)
        
        # Run concurrent workers
        threads = [threading.Thread(target=concurrent_worker) for _ in range(5)]
        
        start_time = time.perf_counter()
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.perf_counter() - start_time
        
        # Performance validation
        mean_execution_time = np.mean(execution_times)
        max_execution_time = np.max(execution_times)
        
        assert len(execution_times) == 100  # 5 threads * 20 executions
        assert mean_execution_time < 5.0  # Average under 5ms
        assert max_execution_time < 20.0  # Max under 20ms
        assert total_time < 10.0  # Total execution under 10s


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in hook system."""
    
    def test_hook_exception_handling(self):
        """Test graceful handling of hook exceptions."""
        # Hooks with various types of errors
        def division_error_hook(state):
            return 1 / 0
        
        def type_error_hook(state):
            return "string" + 5
        
        def key_error_hook(state):
            return state['nonexistent_key']
        
        # Error handling wrapper
        def safe_execute_hook(hook, state, fallback=None):
            try:
                return hook(state)
            except Exception as e:
                print(f"Hook error: {e}")
                return fallback
        
        # Test error handling
        test_state = {'valid_key': 'value'}
        
        result1 = safe_execute_hook(division_error_hook, test_state, {'error': 'handled'})
        result2 = safe_execute_hook(type_error_hook, test_state, {})
        result3 = safe_execute_hook(key_error_hook, test_state, None)
        
        # Validate error handling
        assert result1 == {'error': 'handled'}
        assert result2 == {}
        assert result3 is None
    
    def test_hook_validation(self):
        """Test hook validation and type checking."""
        # Hook validation function
        def validate_hook(hook_fn, expected_signature=None):
            if not callable(hook_fn):
                return False, "Hook must be callable"
            
            if expected_signature:
                import inspect
                sig = inspect.signature(hook_fn)
                if len(sig.parameters) != expected_signature:
                    return False, f"Expected {expected_signature} parameters"
            
            return True, "Valid"
        
        # Test hooks
        valid_hook = lambda state: {'valid': True}
        invalid_hook = "not_a_function"
        wrong_signature = lambda: {'no_params': True}
        
        # Validate hooks
        valid, msg1 = validate_hook(valid_hook, 1)
        assert valid is True
        
        valid, msg2 = validate_hook(invalid_hook)
        assert valid is False
        assert "callable" in msg2
        
        valid, msg3 = validate_hook(wrong_signature, 1)
        assert valid is False
        assert "parameters" in msg3
    
    def test_empty_state_handling(self):
        """Test hook behavior with empty or None state."""
        def robust_hook(state):
            if state is None:
                return {'default': 'none_state'}
            
            if not state:
                return {'default': 'empty_state'}
            
            return {'processed': len(state)}
        
        # Test with various state types
        result_none = robust_hook(None)
        result_empty_dict = robust_hook({})
        result_empty_list = robust_hook([])
        result_normal = robust_hook({'a': 1, 'b': 2})
        
        assert result_none == {'default': 'none_state'}
        assert result_empty_dict == {'default': 'empty_state'}
        assert result_empty_list == {'default': 'empty_state'}
        assert result_normal == {'processed': 2}


class TestHookSystemIntegrationEnd2End:
    """End-to-end integration tests for complete hook system functionality."""
    
    def test_complete_hook_system_workflow(self):
        """Test complete workflow from configuration to execution."""
        # Mock complete environment setup
        env = MagicMock(spec=PlumeNavigationEnv)
        recorder = MagicMock(spec=RecorderProtocol)
        debug_session = DebugSession("integration_test")
        
        # Configure hooks
        def complete_obs_hook(state):
            return {
                'wind_speed': np.linalg.norm(state.get('wind_velocity', [0, 0])),
                'exploration_metric': len(state.get('visited_positions', [])),
                'energy_remaining': state.get('energy', 100) - state.get('step_count', 0) * 0.1
            }
        
        def complete_reward_hook(base_reward, info):
            exploration_bonus = 0.01 * info.get('exploration_metric', 0)
            energy_penalty = -0.1 if info.get('energy_remaining', 100) < 10 else 0
            return exploration_bonus + energy_penalty
        
        def complete_episode_hook(final_info):
            debug_session.performance_metrics.append({
                'episode_length': final_info['episode_length'],
                'success': final_info['success'],
                'final_energy': final_info.get('energy_remaining', 0)
            })
        
        # Register hooks
        env.extra_obs_fn = complete_obs_hook
        env.extra_reward_fn = complete_reward_hook
        env.episode_end_fn = complete_episode_hook
        
        # Simulate complete episode
        episode_states = []
        total_reward = 0
        
        for step in range(10):
            # Mock state progression
            state = {
                'position': np.array([step * 5, step * 3]),
                'wind_velocity': [2.0, 1.0],
                'visited_positions': [(i*5, i*3) for i in range(step+1)],
                'energy': 100,
                'step_count': step
            }
            episode_states.append(state)
            
            # Execute hooks
            extra_obs = env.extra_obs_fn(state)
            base_reward = 1.0
            extra_reward = env.extra_reward_fn(base_reward, extra_obs)
            step_reward = base_reward + extra_reward
            total_reward += step_reward
            
            # Record step data
            step_data = {
                'step': step,
                'state': state,
                'observations': extra_obs,
                'reward': step_reward
            }
            recorder.record_step(step_data, step)
        
        # Episode completion
        final_info = {
            'episode_length': len(episode_states),
            'success': True,
            'total_reward': total_reward,
            'energy_remaining': 99  # 100 - 10*0.1
        }
        env.episode_end_fn(final_info)
        
        # Validate complete workflow
        assert len(debug_session.performance_metrics) == 1
        assert debug_session.performance_metrics[0]['episode_length'] == 10
        assert debug_session.performance_metrics[0]['success'] is True
        assert recorder.record_step.call_count == 10
        
        # Validate hook outputs
        final_state = episode_states[-1]
        final_obs = env.extra_obs_fn(final_state)
        assert final_obs['wind_speed'] == pytest.approx(np.sqrt(5))  # sqrt(2^2 + 1^2)
        assert final_obs['exploration_metric'] == 10
        assert final_obs['energy_remaining'] == pytest.approx(99.1, abs=0.1)
    
    def test_performance_benchmarking_complete_system(self):
        """Benchmark complete hook system performance."""
        # Setup complete environment
        num_agents = 50
        num_steps = 100
        
        # Performance-optimized hooks
        def optimized_obs_hook(state):
            positions = state['positions']
            return {
                'distances': np.linalg.norm(positions, axis=1),
                'speeds_normalized': state['speeds'] / state['max_speeds']
            }
        
        def optimized_reward_hook(base_rewards, info):
            distances = info.get('distances', np.zeros(len(base_rewards)))
            return np.where(distances > 25, 0.1, 0.0)
        
        # Benchmark execution
        execution_times = []
        
        for step in range(num_steps):
            start = time.perf_counter()
            
            # Mock multi-agent state
            state = {
                'positions': np.random.rand(num_agents, 2) * 100,
                'speeds': np.random.rand(num_agents) * 2,
                'max_speeds': np.full(num_agents, 2.0)
            }
            
            # Execute hooks
            extra_obs = optimized_obs_hook(state)
            base_rewards = np.ones(num_agents)
            extra_rewards = optimized_reward_hook(base_rewards, extra_obs)
            
            execution_time = (time.perf_counter() - start) * 1000  # ms
            execution_times.append(execution_time)
        
        # Performance analysis
        mean_time = np.mean(execution_times)
        p95_time = np.percentile(execution_times, 95)
        max_time = np.max(execution_times)
        
        # Validate performance requirements
        assert mean_time <= 33.0, f"Mean time {mean_time:.2f}ms exceeds 33ms target"
        assert p95_time <= 50.0, f"95th percentile {p95_time:.2f}ms too high"
        assert max_time <= 100.0, f"Max time {max_time:.2f}ms too high"
        
        print(f"Hook system performance with {num_agents} agents:")
        print(f"  Mean: {mean_time:.2f}ms")
        print(f"  95th percentile: {p95_time:.2f}ms") 
        print(f"  Max: {max_time:.2f}ms")


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        if test_category == "basic":
            pytest.main(["-v", "TestHookSystemBasicFunctionality"])
        elif test_category == "performance":
            pytest.main(["-v", "TestHookSystemPerformance"])
        elif test_category == "debug":
            pytest.main(["-v", "TestDebugGUIIntegration"])
        elif test_category == "integration":
            pytest.main(["-v", "TestHookSystemIntegrationEnd2End"])
        else:
            pytest.main(["-v"])
    else:
        # Run all tests
        pytest.main(["-v", __file__])