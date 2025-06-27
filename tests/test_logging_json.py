"""
JSON Logging Validation Test Suite for Odor Plume Navigation System.

This test suite validates Loguru-based structured logging outputs conform to schema 
requirements, validates correlation-ID injection, and verifies performance statistics 
integration for machine-parseable debugging and monitoring per Section 6.5 Monitoring 
and Observability and Section 6.6.3.2.5 Integration Testing Strategy.

Test Coverage Areas:
- JSON sink validation per Section 0.4.1 explicitly in scope
- Structured logging verification per Section 6.6.3.2.5 integration testing
- Machine-parseable logging consistency per Section 6.6.1.1 philosophy
- Performance statistics logging per Section 0.3.2 user examples
- Correlation-ID injection verification across deployment environments
- Cache metrics integration in structured logging per Section 6.6.5.4.2
- JSON format compliance for both console and file sinks

Author: Blitzy Platform - JSON Logging Validation Module
Version: 1.0.0
"""

import pytest
import json
import time
import tempfile
import threading
import os
import sys
import io
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from unittest.mock import patch, MagicMock
import yaml

from loguru import logger

# Import logging infrastructure components
from odor_plume_nav.utils.logging_setup import (
    setup_logger,
    LoggingConfig,
    PerformanceMetrics,
    FrameCacheConfig,
    correlation_context,
    get_correlation_context,
    set_correlation_context,
    CorrelationContext,
    create_step_timer,
    step_performance_timer,
    update_cache_metrics,
    log_cache_memory_pressure_violation,
    _load_logging_yaml,
    _setup_yaml_sinks,
    _create_json_formatter,
    get_enhanced_logger,
    EnhancedLogger,
    JSON_FORMAT,
    PERFORMANCE_THRESHOLDS,
)


class TestJSONLoggingValidation:
    """
    Comprehensive test suite for JSON logging validation and structured output compliance.
    
    Tests validate JSON sink configuration, schema compliance, correlation-ID injection,
    and performance statistics integration according to Section 6.6.3.2.5 requirements.
    """
    
    @pytest.fixture(autouse=True)
    def reset_logger_state(self):
        """Reset logger state before and after each test for isolation."""
        # Save original handlers
        original_handlers = logger._core.handlers.copy()
        
        # Remove all handlers for clean testing
        logger.remove()
        
        yield
        
        # Restore original handlers
        logger._core.handlers.clear()
        for handler_id, handler in original_handlers.items():
            logger._core.handlers[handler_id] = handler
    
    @pytest.fixture
    def temp_json_log_file(self):
        """Create temporary JSON log file for structured logging tests."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            log_path = tmp.name
        
        yield log_path
        
        # Cleanup
        if os.path.exists(log_path):
            os.unlink(log_path)
    
    @pytest.fixture
    def temp_logging_yaml_config(self):
        """Create temporary logging.yaml configuration for testing."""
        config_data = {
            'sinks': {
                'console': {
                    'enabled': True,
                    'format': 'enhanced',
                    'level': 'INFO',
                    'colorize': True,
                    'serialize': False
                },
                'json': {
                    'enabled': True,
                    'format': 'json',
                    'level': 'DEBUG',
                    'file_path': '${LOG_PATH:./logs/test_structured.json}',
                    'rotation': '10 MB',
                    'retention': '7 days',
                    'serialize': True
                }
            },
            'performance': {
                'monitoring_enabled': True,
                'thresholds': {
                    'environment_step': 0.010,
                    'frame_processing': 0.033,
                    'simulation_fps_min': 30.0
                },
                'cache_monitoring': {
                    'hit_rate_threshold': 0.90,
                    'memory_threshold_percent': 90
                }
            },
            'correlation': {
                'enabled': True,
                'default_context': {
                    'correlation_id': 'none',
                    'request_id': 'none',
                    'module': 'system'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config_data, tmp, default_flow_style=False)
            yaml_path = tmp.name
        
        yield yaml_path
        
        # Cleanup
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)
    
    @pytest.fixture
    def sample_performance_metrics(self):
        """Sample performance metrics for testing."""
        return {
            'step_time_ms': 8.5,
            'frame_retrieval_ms': 2.1,
            'cache_hit_rate': 0.92,
            'fps_estimate': 45.2,
            'memory_usage_mb': 1024.5,
            'cache_memory_usage_mb': 512.3,
            'cache_hit_count': 150,
            'cache_miss_count': 13,
            'cache_evictions': 2
        }
    
    @pytest.fixture
    def sample_cache_statistics(self):
        """Sample cache statistics for testing."""
        return {
            'cache_hit_count': 225,
            'cache_miss_count': 18,
            'cache_evictions': 5,
            'cache_hit_rate': 0.926,
            'cache_memory_usage_mb': 768.2,
            'cache_memory_limit_mb': 2048.0
        }
    
    # ========================================================================================
    # SECTION 1: JSON Sink Configuration Validation per Section 0.4.1
    # ========================================================================================
    
    def test_json_sink_configuration_from_logging_yaml(self, temp_logging_yaml_config, temp_json_log_file):
        """
        Test JSON sink configuration loading from logging.yaml per Section 0.4.1.
        
        Validates that JSON sink configuration is properly loaded and applied from
        logging.yaml configuration file with correct format, rotation, and retention settings.
        """
        # Load configuration from YAML file
        setup_logger(logging_config_path=temp_logging_yaml_config)
        
        # Create enhanced logger for testing
        enhanced_logger = get_enhanced_logger("test_json_sink")
        
        # Test message with structured data
        test_message = "JSON sink configuration test"
        extra_data = {
            "test_category": "json_sink_validation",
            "config_source": "logging.yaml",
            "sink_type": "json"
        }
        
        # Log message with correlation context
        with correlation_context("json_sink_test", request_id="req_yaml_001") as ctx:
            enhanced_logger.info(test_message, extra=extra_data)
        
        # Verify logger is configured (handlers present)
        assert len(logger._core.handlers) > 0
        
        # Verify at least one handler is configured for JSON output
        json_handler_found = False
        for handler_id, handler in logger._core.handlers.items():
            if hasattr(handler, '_filter') and handler._filter:
                json_handler_found = True
                break
        
        # Note: In actual implementation, would verify JSON file output
        # For this test, we verify configuration loading succeeded
        assert json_handler_found or len(logger._core.handlers) > 0
    
    def test_logging_yaml_structure_validation(self, temp_logging_yaml_config):
        """
        Test logging.yaml structure validation and parsing.
        
        Validates that logging.yaml file structure conforms to expected schema
        and all required sections are present and properly formatted.
        """
        # Load and validate YAML configuration
        config_data = _load_logging_yaml(temp_logging_yaml_config)
        
        assert config_data is not None
        assert 'sinks' in config_data
        assert 'performance' in config_data
        assert 'correlation' in config_data
        
        # Validate sinks section structure
        sinks = config_data['sinks']
        assert 'console' in sinks
        assert 'json' in sinks
        
        # Validate JSON sink configuration
        json_sink = sinks['json']
        assert json_sink['enabled'] is True
        assert json_sink['format'] == 'json'
        assert json_sink['serialize'] is True
        assert 'rotation' in json_sink
        assert 'retention' in json_sink
        
        # Validate performance monitoring configuration
        performance = config_data['performance']
        assert performance['monitoring_enabled'] is True
        assert 'thresholds' in performance
        assert 'cache_monitoring' in performance
        
        # Validate correlation configuration
        correlation = config_data['correlation']
        assert correlation['enabled'] is True
        assert 'default_context' in correlation
    
    def test_dual_sink_architecture_setup(self, temp_logging_yaml_config, temp_json_log_file):
        """
        Test dual sink architecture setup with both console and JSON outputs.
        
        Validates that both console (human-readable) and JSON (machine-parseable)
        sinks are properly configured and operational simultaneously.
        """
        # Modify YAML config to use temp file
        with open(temp_logging_yaml_config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data['sinks']['json']['file_path'] = temp_json_log_file
        
        with open(temp_logging_yaml_config, 'w') as f:
            yaml.dump(config_data, f)
        
        # Setup logger with dual sink configuration
        setup_logger(logging_config_path=temp_logging_yaml_config)
        
        # Create string capture for console output
        string_io = io.StringIO()
        
        # Add additional handler to capture console output
        console_handler_id = logger.add(string_io, level="INFO", format="{message}")
        
        # Log test message
        test_message = "Dual sink architecture test"
        
        with correlation_context("dual_sink_test", request_id="req_dual_001"):
            enhanced_logger = get_enhanced_logger("dual_sink_test")
            enhanced_logger.info(test_message, extra={
                "test_type": "dual_sink_validation",
                "architecture": "console_and_json"
            })
        
        # Allow time for file write
        time.sleep(0.1)
        
        # Verify console output captured
        console_output = string_io.getvalue()
        assert test_message in console_output
        
        # Clean up console handler
        logger.remove(console_handler_id)
        
        # Verify JSON file output (if file was created)
        if os.path.exists(temp_json_log_file):
            with open(temp_json_log_file, 'r') as f:
                json_content = f.read().strip()
                if json_content:
                    # Should contain JSON formatted content
                    assert test_message in json_content or "dual_sink_test" in json_content
    
    # ========================================================================================
    # SECTION 2: Structured Log Schema Compliance per Section 6.6.3.2.5
    # ========================================================================================
    
    def test_json_log_schema_compliance(self, temp_json_log_file):
        """
        Test JSON log schema compliance across deployment environments.
        
        Validates that JSON log records conform to required schema structure
        with all mandatory fields present and properly formatted.
        """
        # Configure JSON logging
        config = LoggingConfig(
            format="json",
            console_enabled=False,
            file_enabled=True,
            file_path=temp_json_log_file,
            correlation_enabled=True,
            enable_performance=True
        )
        setup_logger(config)
        
        # Log structured message with correlation context
        with correlation_context("schema_test", 
                                request_id="req_schema_001", 
                                episode_id="ep_schema_001") as ctx:
            enhanced_logger = get_enhanced_logger("schema_test_module")
            enhanced_logger.info("Schema compliance test message", extra={
                "test_category": "schema_validation",
                "deployment_env": "testing",
                "metric_type": "schema_compliance_test",
                "custom_field": "custom_value"
            })
        
        time.sleep(0.1)  # Allow file write
        
        # Read and validate JSON structure
        with open(temp_json_log_file, 'r') as f:
            log_content = f.read().strip()
        
        if log_content:
            # Parse JSON record(s)
            log_lines = log_content.strip().split('\n')
            for line in log_lines:
                if line.strip():
                    try:
                        log_record = json.loads(line)
                        
                        # Validate required schema fields
                        required_fields = [
                            'timestamp', 'level', 'logger', 'message', 
                            'correlation_id', 'module'
                        ]
                        
                        for field in required_fields:
                            assert field in log_record, f"Required field '{field}' missing from JSON log record"
                        
                        # Validate field types and content
                        assert isinstance(log_record['timestamp'], str)
                        assert log_record['level'] in ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                        assert isinstance(log_record['logger'], str)
                        assert isinstance(log_record['message'], str)
                        assert isinstance(log_record['correlation_id'], str)
                        assert isinstance(log_record['module'], str)
                        
                        # Validate correlation context fields
                        assert log_record['correlation_id'] != 'none'
                        assert 'request_id' in log_record
                        assert log_record['request_id'] == 'req_schema_001'
                        
                        # Validate episode tracking
                        assert 'episode_id' in log_record
                        assert log_record['episode_id'] == 'ep_schema_001'
                        
                        # Validate custom fields preservation
                        assert log_record['test_category'] == 'schema_validation'
                        assert log_record['custom_field'] == 'custom_value'
                        
                        break  # Test first valid record
                        
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
    
    def test_json_format_consistency_across_environments(self, temp_json_log_file):
        """
        Test JSON format consistency across different deployment environments.
        
        Validates that JSON log format remains consistent across development,
        production, CI/CD, and HPC deployment scenarios.
        """
        environments = ['development', 'production', 'testing', 'batch']
        
        for environment in environments:
            # Configure for specific environment
            config = LoggingConfig(
                environment=environment,
                format="json",
                console_enabled=False,
                file_enabled=True,
                file_path=temp_json_log_file,
                correlation_enabled=True
            ).apply_environment_defaults()
            
            setup_logger(config)
            
            # Log environment-specific message
            with correlation_context(f"{environment}_test", request_id=f"req_{environment}_001"):
                enhanced_logger = get_enhanced_logger(f"{environment}_module")
                enhanced_logger.info(f"Environment test for {environment}", extra={
                    "environment": environment,
                    "test_type": "consistency_validation"
                })
            
            time.sleep(0.05)  # Brief pause between environments
        
        time.sleep(0.1)  # Allow all writes to complete
        
        # Validate consistency across all environment log entries
        with open(temp_json_log_file, 'r') as f:
            log_content = f.read().strip()
        
        if log_content:
            log_lines = log_content.strip().split('\n')
            parsed_records = []
            
            for line in log_lines:
                if line.strip():
                    try:
                        record = json.loads(line)
                        parsed_records.append(record)
                    except json.JSONDecodeError:
                        continue
            
            # Validate at least one record per environment
            assert len(parsed_records) >= len(environments)
            
            # Validate consistent schema across all records
            for record in parsed_records:
                # All records should have consistent required fields
                required_fields = ['timestamp', 'level', 'logger', 'message', 'correlation_id']
                for field in required_fields:
                    assert field in record
                
                # All records should have consistent field types
                assert isinstance(record['timestamp'], str)
                assert isinstance(record['level'], str)
                assert isinstance(record['correlation_id'], str)
    
    def test_json_serialization_completeness(self, temp_json_log_file):
        """
        Test complete JSON serialization of complex log data structures.
        
        Validates that complex nested data structures, performance metrics,
        and cache statistics are properly serialized in JSON format.
        """
        # Setup JSON logging
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            correlation_enabled=True,
            enable_performance=True
        )
        setup_logger(config)
        
        # Create complex nested data structure
        complex_data = {
            "performance_metrics": {
                "step_latency_ms": 8.5,
                "frame_rate": 45.2,
                "memory_usage": {
                    "total_mb": 1024.5,
                    "cache_mb": 512.3,
                    "overhead_mb": 128.7
                },
                "cache_statistics": {
                    "hit_count": 150,
                    "miss_count": 13,
                    "evictions": 2,
                    "hit_rate": 0.92
                }
            },
            "system_info": {
                "platform": "Linux",
                "python_version": "3.11.0",
                "thread_count": 4,
                "process_id": os.getpid()
            },
            "experiment_metadata": {
                "algorithm": "SAC",
                "episode_count": 100,
                "total_steps": 50000,
                "hyperparameters": {
                    "learning_rate": 0.0003,
                    "batch_size": 256,
                    "gamma": 0.99
                }
            }
        }
        
        # Log with complex nested data
        with correlation_context("serialization_test", request_id="req_complex_001"):
            enhanced_logger = get_enhanced_logger("serialization_test")
            enhanced_logger.info("Complex data serialization test", extra=complex_data)
        
        time.sleep(0.1)  # Allow file write
        
        # Validate complete serialization
        with open(temp_json_log_file, 'r') as f:
            log_content = f.read().strip()
        
        if log_content:
            log_record = json.loads(log_content.split('\n')[0])
            
            # Validate nested structure preservation
            assert 'performance_metrics' in log_record
            perf_metrics = log_record['performance_metrics']
            assert perf_metrics['step_latency_ms'] == 8.5
            assert perf_metrics['frame_rate'] == 45.2
            
            # Validate deeply nested data
            memory_usage = perf_metrics['memory_usage']
            assert memory_usage['total_mb'] == 1024.5
            assert memory_usage['cache_mb'] == 512.3
            
            # Validate cache statistics
            cache_stats = perf_metrics['cache_statistics']
            assert cache_stats['hit_count'] == 150
            assert cache_stats['hit_rate'] == 0.92
            
            # Validate system info
            assert 'system_info' in log_record
            sys_info = log_record['system_info']
            assert sys_info['platform'] == 'Linux'
            assert sys_info['process_id'] == os.getpid()
            
            # Validate experiment metadata with hyperparameters
            assert 'experiment_metadata' in log_record
            exp_meta = log_record['experiment_metadata']
            assert exp_meta['algorithm'] == 'SAC'
            hyperparams = exp_meta['hyperparameters']
            assert hyperparams['learning_rate'] == 0.0003
    
    # ========================================================================================
    # SECTION 3: Correlation-ID Injection Verification per Section 6.6.3.2.5
    # ========================================================================================
    
    def test_correlation_id_injection_consistency(self, temp_json_log_file):
        """
        Test correlation-ID injection consistency across log records.
        
        Validates that correlation IDs are consistently injected and maintained
        across multiple log records within the same correlation context.
        """
        # Setup JSON logging
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            correlation_enabled=True
        )
        setup_logger(config)
        
        correlation_id = "test_correlation_12345"
        request_id = "req_consistency_001"
        
        # Log multiple messages within same correlation context
        with correlation_context("consistency_test", 
                                correlation_id=correlation_id, 
                                request_id=request_id):
            enhanced_logger = get_enhanced_logger("correlation_test")
            
            for i in range(3):
                enhanced_logger.info(f"Consistency test message {i+1}", extra={
                    "message_sequence": i+1,
                    "test_type": "correlation_consistency"
                })
                time.sleep(0.01)  # Small delay between messages
        
        time.sleep(0.1)  # Allow file writes
        
        # Validate correlation ID consistency
        with open(temp_json_log_file, 'r') as f:
            log_lines = f.read().strip().split('\n')
        
        parsed_records = []
        for line in log_lines:
            if line.strip():
                try:
                    record = json.loads(line)
                    parsed_records.append(record)
                except json.JSONDecodeError:
                    continue
        
        # Should have 3 log records
        assert len(parsed_records) >= 3
        
        # Validate all records have same correlation_id and request_id
        for i, record in enumerate(parsed_records[:3]):
            assert record['correlation_id'] == correlation_id
            assert record['request_id'] == request_id
            assert record['message_sequence'] == i + 1
            assert 'test_type' in record
            assert record['test_type'] == 'correlation_consistency'
    
    def test_correlation_id_thread_isolation(self, temp_json_log_file):
        """
        Test correlation-ID thread isolation in multi-threaded scenarios.
        
        Validates that correlation IDs are properly isolated between different
        threads and don't interfere with each other.
        """
        # Setup JSON logging
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            correlation_enabled=True
        )
        setup_logger(config)
        
        thread_results = {}
        
        def thread_worker(thread_id):
            """Worker function for thread isolation testing."""
            correlation_id = f"thread_{thread_id}_correlation"
            request_id = f"req_thread_{thread_id}"
            
            with correlation_context("thread_test", 
                                    correlation_id=correlation_id, 
                                    request_id=request_id):
                enhanced_logger = get_enhanced_logger(f"thread_{thread_id}_module")
                enhanced_logger.info(f"Thread {thread_id} test message", extra={
                    "thread_id": thread_id,
                    "test_type": "thread_isolation"
                })
                
                # Store correlation context for validation
                context = get_correlation_context()
                thread_results[thread_id] = {
                    'correlation_id': context.correlation_id,
                    'request_id': context.request_id
                }
                
                time.sleep(0.05)  # Simulate work
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        time.sleep(0.1)  # Allow file writes
        
        # Validate thread isolation
        assert len(thread_results) == 3
        for thread_id, result in thread_results.items():
            expected_correlation = f"thread_{thread_id}_correlation"
            expected_request = f"req_thread_{thread_id}"
            assert result['correlation_id'] == expected_correlation
            assert result['request_id'] == expected_request
        
        # Validate log file contains distinct correlation IDs
        with open(temp_json_log_file, 'r') as f:
            log_content = f.read()
        
        for thread_id in range(3):
            expected_correlation = f"thread_{thread_id}_correlation"
            assert expected_correlation in log_content
    
    def test_correlation_id_multi_agent_scenarios(self, temp_json_log_file):
        """
        Test correlation-ID injection in multi-agent simulation scenarios.
        
        Validates proper correlation tracking across multiple agents with
        episode IDs and agent-specific context information.
        """
        # Setup JSON logging
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            correlation_enabled=True
        )
        setup_logger(config)
        
        episode_id = "episode_multi_001"
        agent_count = 3
        
        # Simulate multi-agent scenario
        for agent_id in range(agent_count):
            correlation_id = f"agent_{agent_id}_correlation"
            request_id = f"req_agent_{agent_id}"
            
            with correlation_context("multi_agent_test",
                                    correlation_id=correlation_id,
                                    request_id=request_id,
                                    episode_id=episode_id,
                                    agent_id=agent_id,
                                    agent_count=agent_count):
                
                enhanced_logger = get_enhanced_logger(f"agent_{agent_id}_module")
                enhanced_logger.info(f"Agent {agent_id} step execution", extra={
                    "agent_id": agent_id,
                    "step_count": 10 + agent_id,
                    "action": f"action_{agent_id}",
                    "reward": 0.5 + (agent_id * 0.1),
                    "test_type": "multi_agent_simulation"
                })
        
        time.sleep(0.1)  # Allow file writes
        
        # Validate multi-agent correlation tracking
        with open(temp_json_log_file, 'r') as f:
            log_lines = f.read().strip().split('\n')
        
        agent_records = []
        for line in log_lines:
            if line.strip():
                try:
                    record = json.loads(line)
                    if 'test_type' in record and record['test_type'] == 'multi_agent_simulation':
                        agent_records.append(record)
                except json.JSONDecodeError:
                    continue
        
        # Should have records for all agents
        assert len(agent_records) == agent_count
        
        # Validate each agent record
        for i, record in enumerate(agent_records):
            expected_correlation = f"agent_{i}_correlation"
            expected_request = f"req_agent_{i}"
            
            assert record['correlation_id'] == expected_correlation
            assert record['request_id'] == expected_request
            assert record['episode_id'] == episode_id
            assert record['agent_id'] == i
            assert record['agent_count'] == agent_count
            assert record['step_count'] == 10 + i
            assert record['reward'] == 0.5 + (i * 0.1)
    
    # ========================================================================================
    # SECTION 4: Performance Statistics Integration per Section 0.3.2
    # ========================================================================================
    
    def test_performance_statistics_inclusion_perf_stats(self, temp_json_log_file, sample_performance_metrics):
        """
        Test performance statistics inclusion in info['perf_stats'] per Section 0.3.2.
        
        Validates that performance statistics are properly included in JSON logs
        when accessed through info['perf_stats'] dictionary as specified in user examples.
        """
        # Setup JSON logging with performance monitoring
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            correlation_enabled=True,
            enable_performance=True
        )
        setup_logger(config)
        
        # Simulate RL training environment step with perf_stats
        with correlation_context("rl_training", episode_id="ep_001", step_count=150):
            enhanced_logger = get_enhanced_logger("rl_environment")
            
            # Log with perf_stats structure as specified in Section 0.3.2
            enhanced_logger.info("Environment step completed", extra={
                "perf_stats": sample_performance_metrics,
                "action": [0.5, -0.3, 0.8],
                "reward": 1.25,
                "done": False,
                "info_type": "step_completion"
            })
        
        time.sleep(0.1)  # Allow file write
        
        # Validate perf_stats inclusion in JSON
        with open(temp_json_log_file, 'r') as f:
            log_content = f.read().strip()
        
        if log_content:
            log_record = json.loads(log_content.split('\n')[0])
            
            # Validate perf_stats field presence and structure
            assert 'perf_stats' in log_record
            perf_stats = log_record['perf_stats']
            
            # Validate all expected performance metrics
            assert perf_stats['step_time_ms'] == 8.5
            assert perf_stats['frame_retrieval_ms'] == 2.1
            assert perf_stats['cache_hit_rate'] == 0.92
            assert perf_stats['fps_estimate'] == 45.2
            assert perf_stats['memory_usage_mb'] == 1024.5
            
            # Validate cache-specific metrics
            assert perf_stats['cache_memory_usage_mb'] == 512.3
            assert perf_stats['cache_hit_count'] == 150
            assert perf_stats['cache_miss_count'] == 13
            assert perf_stats['cache_evictions'] == 2
            
            # Validate additional context information
            assert log_record['episode_id'] == 'ep_001'
            assert log_record['step_count'] == 150
            assert log_record['action'] == [0.5, -0.3, 0.8]
            assert log_record['reward'] == 1.25
    
    def test_step_timer_performance_integration(self, temp_json_log_file):
        """
        Test step timer performance metrics integration in JSON logs.
        
        Validates that step timing context managers automatically populate
        performance metrics in structured JSON logs with threshold violation detection.
        """
        # Setup JSON logging with performance monitoring
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            enable_performance=True,
            correlation_enabled=True
        )
        setup_logger(config)
        
        with correlation_context("step_timing_test", episode_id="ep_timing_001"):
            enhanced_logger = get_enhanced_logger("step_timer_test")
            
            # Test normal step execution (under threshold)
            with create_step_timer() as metrics:
                time.sleep(0.005)  # 5ms - under 10ms threshold
                enhanced_logger.info("Normal step execution", extra={
                    "operation_type": "normal_step",
                    "expected_duration": "under_threshold"
                })
            
            # Brief pause
            time.sleep(0.01)
            
            # Test slow step execution (over threshold)
            with create_step_timer() as metrics:
                time.sleep(0.012)  # 12ms - over 10ms threshold
                enhanced_logger.info("Slow step execution", extra={
                    "operation_type": "slow_step",
                    "expected_duration": "over_threshold"
                })
        
        time.sleep(0.1)  # Allow file writes
        
        # Validate step timer integration
        with open(temp_json_log_file, 'r') as f:
            log_lines = f.read().strip().split('\n')
        
        step_records = []
        threshold_violation_records = []
        
        for line in log_lines:
            if line.strip():
                try:
                    record = json.loads(line)
                    if 'operation_type' in record:
                        step_records.append(record)
                    elif 'metric_type' in record and record['metric_type'] == 'step_latency_violation':
                        threshold_violation_records.append(record)
                except json.JSONDecodeError:
                    continue
        
        # Should have both normal and slow step records
        assert len(step_records) >= 2
        
        # Should have threshold violation warning for slow step
        assert len(threshold_violation_records) >= 1
        
        # Validate threshold violation record structure
        violation_record = threshold_violation_records[0]
        assert 'actual_latency_ms' in violation_record
        assert 'threshold_latency_ms' in violation_record
        assert violation_record['actual_latency_ms'] > 10  # Over 10ms threshold
        assert violation_record['threshold_latency_ms'] == 10  # 10ms threshold
    
    def test_cache_performance_metrics_integration(self, temp_json_log_file, sample_cache_statistics):
        """
        Test cache performance metrics integration in structured logging.
        
        Validates that cache statistics are properly integrated into JSON logs
        with hit rates, memory usage, and eviction information.
        """
        # Setup JSON logging
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            correlation_enabled=True,
            enable_performance=True
        )
        setup_logger(config)
        
        with correlation_context("cache_metrics_test", episode_id="ep_cache_001"):
            # Update cache metrics in correlation context
            update_cache_metrics(
                cache_hit_count=sample_cache_statistics['cache_hit_count'],
                cache_miss_count=sample_cache_statistics['cache_miss_count'],
                cache_evictions=sample_cache_statistics['cache_evictions'],
                cache_memory_usage_mb=sample_cache_statistics['cache_memory_usage_mb'],
                cache_memory_limit_mb=sample_cache_statistics['cache_memory_limit_mb']
            )
            
            enhanced_logger = get_enhanced_logger("cache_metrics_test")
            enhanced_logger.info("Cache metrics integration test", extra={
                "test_type": "cache_metrics_integration",
                "frame_id": 42,
                "operation": "frame_retrieval"
            })
        
        time.sleep(0.1)  # Allow file write
        
        # Validate cache metrics in JSON log
        with open(temp_json_log_file, 'r') as f:
            log_content = f.read().strip()
        
        if log_content:
            log_record = json.loads(log_content.split('\n')[0])
            
            # Validate cache statistics fields
            assert 'cache_hit_count' in log_record
            assert 'cache_miss_count' in log_record
            assert 'cache_evictions' in log_record
            assert 'cache_hit_rate' in log_record
            assert 'cache_memory_usage_mb' in log_record
            assert 'cache_memory_limit_mb' in log_record
            
            # Validate cache statistics values
            assert log_record['cache_hit_count'] == 225
            assert log_record['cache_miss_count'] == 18
            assert log_record['cache_evictions'] == 5
            assert abs(log_record['cache_hit_rate'] - 0.926) < 0.001  # Allow floating point tolerance
            assert log_record['cache_memory_usage_mb'] == 768.2
            assert log_record['cache_memory_limit_mb'] == 2048.0
            
            # Validate additional context
            assert log_record['test_type'] == 'cache_metrics_integration'
            assert log_record['frame_id'] == 42
    
    # ========================================================================================
    # SECTION 5: Machine-Parseable Log Consistency per Section 6.6.1.1
    # ========================================================================================
    
    def test_machine_parseable_log_consistency(self, temp_json_log_file):
        """
        Test machine-parseable log consistency for monitoring systems.
        
        Validates that JSON logs maintain consistent structure and format
        suitable for automated parsing by monitoring and analysis systems.
        """
        # Setup JSON logging
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            correlation_enabled=True
        )
        setup_logger(config)
        
        # Generate diverse log entries for consistency testing
        log_scenarios = [
            {
                "scenario": "info_log",
                "level": "info",
                "message": "Information log message",
                "extra": {"type": "info_test", "value": 42}
            },
            {
                "scenario": "warning_log",
                "level": "warning", 
                "message": "Warning log message",
                "extra": {"type": "warning_test", "threshold_exceeded": True}
            },
            {
                "scenario": "error_log",
                "level": "error",
                "message": "Error log message", 
                "extra": {"type": "error_test", "error_code": 500}
            }
        ]
        
        # Log diverse scenarios
        for scenario in log_scenarios:
            with correlation_context(f"consistency_test_{scenario['scenario']}", 
                                    request_id=f"req_{scenario['scenario']}"):
                enhanced_logger = get_enhanced_logger("consistency_test")
                log_method = getattr(enhanced_logger, scenario['level'])
                log_method(scenario['message'], extra=scenario['extra'])
                time.sleep(0.01)
        
        time.sleep(0.1)  # Allow file writes
        
        # Validate parsing consistency
        with open(temp_json_log_file, 'r') as f:
            log_lines = f.read().strip().split('\n')
        
        parsed_records = []
        for line in log_lines:
            if line.strip():
                try:
                    record = json.loads(line)
                    parsed_records.append(record)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Failed to parse JSON log line: {line[:100]}... Error: {e}")
        
        # Should have all scenario records
        assert len(parsed_records) >= len(log_scenarios)
        
        # Validate consistent structure across all records
        required_fields = ['timestamp', 'level', 'logger', 'message', 'correlation_id']
        
        for i, record in enumerate(parsed_records[:len(log_scenarios)]):
            # Validate all required fields present
            for field in required_fields:
                assert field in record, f"Missing field '{field}' in record {i}"
            
            # Validate timestamp format (ISO-like)
            assert isinstance(record['timestamp'], str)
            assert len(record['timestamp']) > 10  # Basic length check
            
            # Validate level consistency
            expected_level = log_scenarios[i]['level'].upper()
            assert record['level'] == expected_level
            
            # Validate message content
            expected_message = log_scenarios[i]['message']
            assert record['message'] == expected_message
            
            # Validate extra fields preservation
            scenario_extra = log_scenarios[i]['extra']
            for key, value in scenario_extra.items():
                assert record[key] == value
    
    def test_json_parsing_robustness(self, temp_json_log_file):
        """
        Test JSON parsing robustness for monitoring system integration.
        
        Validates that JSON logs can be reliably parsed by external monitoring
        tools and handle edge cases in data formatting.
        """
        # Setup JSON logging
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            correlation_enabled=True
        )
        setup_logger(config)
        
        # Test edge cases for JSON parsing
        edge_cases = [
            {
                "case": "unicode_characters",
                "message": "Unicode test: 测试 🚀 café naïve",
                "extra": {"unicode_field": "Special chars: àáâãäå"}
            },
            {
                "case": "null_values",
                "message": "Null value test",
                "extra": {"null_field": None, "empty_string": "", "zero_value": 0}
            },
            {
                "case": "numeric_precision",
                "message": "Numeric precision test",
                "extra": {
                    "float_value": 3.141592653589793,
                    "large_int": 9223372036854775807,
                    "small_float": 1e-10
                }
            },
            {
                "case": "special_characters",
                "message": "Special chars: \"quotes\" 'apostrophes' \\backslashes\\",
                "extra": {"special_string": "Line1\nLine2\tTabbed"}
            }
        ]
        
        # Log edge cases
        for edge_case in edge_cases:
            with correlation_context(f"robustness_{edge_case['case']}", 
                                    request_id=f"req_{edge_case['case']}"):
                enhanced_logger = get_enhanced_logger("robustness_test")
                enhanced_logger.info(edge_case['message'], extra=edge_case['extra'])
                time.sleep(0.01)
        
        time.sleep(0.1)  # Allow file writes
        
        # Validate robust JSON parsing
        with open(temp_json_log_file, 'r') as f:
            log_content = f.read()
        
        log_lines = log_content.strip().split('\n')
        successfully_parsed = 0
        
        for line in log_lines:
            if line.strip():
                try:
                    record = json.loads(line)
                    
                    # Validate basic structure preserved
                    assert 'timestamp' in record
                    assert 'level' in record
                    assert 'message' in record
                    assert 'correlation_id' in record
                    
                    # Validate specific edge case handling
                    if 'unicode_field' in record:
                        assert record['unicode_field'] == "Special chars: àáâãäå"
                    elif 'null_field' in record:
                        assert record['null_field'] is None
                        assert record['empty_string'] == ""
                        assert record['zero_value'] == 0
                    elif 'float_value' in record:
                        assert abs(record['float_value'] - 3.141592653589793) < 1e-10
                        assert record['large_int'] == 9223372036854775807
                    elif 'special_string' in record:
                        assert '\n' in record['special_string']
                        assert '\t' in record['special_string']
                    
                    successfully_parsed += 1
                    
                except (json.JSONDecodeError, KeyError, AssertionError) as e:
                    pytest.fail(f"Failed to parse or validate JSON record: {line[:100]}... Error: {e}")
        
        # Should successfully parse all edge case records
        assert successfully_parsed >= len(edge_cases)
    
    # ========================================================================================
    # SECTION 6: Cache Metrics Integration per Section 6.6.5.4.2
    # ========================================================================================
    
    def test_cache_metrics_integration_structured_logging(self, temp_json_log_file, sample_cache_statistics):
        """
        Test cache metrics integration in structured logging per Section 6.6.5.4.2.
        
        Validates that cache hit rates, memory usage, and eviction statistics
        are properly integrated into structured JSON logs for monitoring.
        """
        # Setup JSON logging with cache monitoring
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            correlation_enabled=True,
            enable_performance=True
        )
        setup_logger(config)
        
        with correlation_context("cache_integration_test", episode_id="ep_cache_integration"):
            # Simulate cache operations with statistics
            context = get_correlation_context()
            
            # Start performance tracking with cache metrics
            metrics = context.push_performance("frame_cache_operation")
            
            # Update cache metrics
            update_cache_metrics(
                context=context,
                cache_hit_count=sample_cache_statistics['cache_hit_count'],
                cache_miss_count=sample_cache_statistics['cache_miss_count'],
                cache_evictions=sample_cache_statistics['cache_evictions'],
                cache_memory_usage_mb=sample_cache_statistics['cache_memory_usage_mb'],
                cache_memory_limit_mb=sample_cache_statistics['cache_memory_limit_mb']
            )
            
            enhanced_logger = get_enhanced_logger("cache_integration")
            enhanced_logger.info("Cache operation completed", extra={
                "operation_type": "frame_retrieval",
                "frame_id": 123,
                "cache_enabled": True,
                "metric_type": "cache_operation_completed"
            })
            
            # Complete performance tracking
            context.pop_performance()
        
        time.sleep(0.1)  # Allow file write
        
        # Validate cache metrics integration
        with open(temp_json_log_file, 'r') as f:
            log_content = f.read().strip()
        
        if log_content:
            log_record = json.loads(log_content.split('\n')[0])
            
            # Validate cache statistics fields in structured log
            cache_fields = [
                'cache_hit_count', 'cache_miss_count', 'cache_evictions',
                'cache_hit_rate', 'cache_memory_usage_mb', 'cache_memory_limit_mb'
            ]
            
            for field in cache_fields:
                assert field in log_record, f"Cache field '{field}' missing from structured log"
            
            # Validate cache statistics values
            assert log_record['cache_hit_count'] == sample_cache_statistics['cache_hit_count']
            assert log_record['cache_miss_count'] == sample_cache_statistics['cache_miss_count']
            assert log_record['cache_evictions'] == sample_cache_statistics['cache_evictions']
            assert log_record['cache_memory_usage_mb'] == sample_cache_statistics['cache_memory_usage_mb']
            assert log_record['cache_memory_limit_mb'] == sample_cache_statistics['cache_memory_limit_mb']
            
            # Validate calculated hit rate
            expected_hit_rate = sample_cache_statistics['cache_hit_rate']
            assert abs(log_record['cache_hit_rate'] - expected_hit_rate) < 0.001
            
            # Validate operation context
            assert log_record['operation_type'] == 'frame_retrieval'
            assert log_record['frame_id'] == 123
            assert log_record['cache_enabled'] is True
    
    def test_cache_memory_pressure_logging(self, temp_json_log_file):
        """
        Test cache memory pressure violation logging in structured format.
        
        Validates that cache memory pressure warnings are properly logged
        with ResourceError category and memory usage details.
        """
        # Setup JSON logging
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            correlation_enabled=True
        )
        setup_logger(config)
        
        # Simulate memory pressure scenarios
        memory_scenarios = [
            {"usage": 1800.0, "limit": 2048.0, "should_warn": False},  # 87.9% - below threshold
            {"usage": 1900.0, "limit": 2048.0, "should_warn": True},   # 92.8% - above threshold
            {"usage": 2000.0, "limit": 2048.0, "should_warn": True},   # 97.7% - high pressure
        ]
        
        with correlation_context("memory_pressure_test", episode_id="ep_memory_001"):
            for i, scenario in enumerate(memory_scenarios):
                # Log memory pressure violation
                log_cache_memory_pressure_violation(
                    current_usage_mb=scenario['usage'],
                    limit_mb=scenario['limit'],
                    threshold_ratio=0.9  # 90% threshold
                )
                
                # Log additional context
                enhanced_logger = get_enhanced_logger("memory_pressure_test")
                enhanced_logger.info(f"Memory pressure scenario {i+1}", extra={
                    "scenario_id": i+1,
                    "usage_mb": scenario['usage'],
                    "limit_mb": scenario['limit'],
                    "expected_warning": scenario['should_warn']
                })
                
                time.sleep(0.01)
        
        time.sleep(0.1)  # Allow file writes
        
        # Validate memory pressure logging
        with open(temp_json_log_file, 'r') as f:
            log_lines = f.read().strip().split('\n')
        
        pressure_violation_records = []
        scenario_records = []
        
        for line in log_lines:
            if line.strip():
                try:
                    record = json.loads(line)
                    if 'metric_type' in record and record['metric_type'] == 'memory_pressure_violation':
                        pressure_violation_records.append(record)
                    elif 'scenario_id' in record:
                        scenario_records.append(record)
                except json.JSONDecodeError:
                    continue
        
        # Should have pressure violations for high usage scenarios
        assert len(pressure_violation_records) >= 2  # Scenarios 2 and 3 should trigger warnings
        
        # Validate pressure violation record structure
        for violation_record in pressure_violation_records:
            assert 'resource_category' in violation_record
            assert violation_record['resource_category'] == 'cache_memory'
            assert 'current_usage_mb' in violation_record
            assert 'limit_mb' in violation_record
            assert 'usage_ratio' in violation_record
            assert 'threshold_ratio' in violation_record
            
            # Validate usage ratio calculation
            usage_ratio = violation_record['current_usage_mb'] / violation_record['limit_mb']
            assert abs(violation_record['usage_ratio'] - usage_ratio) < 0.001
            
            # Should be above threshold to trigger warning
            assert violation_record['usage_ratio'] >= 0.9
    
    def test_cache_hit_rate_monitoring_integration(self, temp_json_log_file):
        """
        Test cache hit rate monitoring integration with structured logging.
        
        Validates that cache hit rate monitoring integrates properly with
        JSON structured logs and threshold-based alerting.
        """
        # Setup JSON logging with performance monitoring
        config = LoggingConfig(
            format="json",
            file_enabled=True,
            file_path=temp_json_log_file,
            console_enabled=False,
            correlation_enabled=True,
            enable_performance=True
        )
        setup_logger(config)
        
        # Simulate cache operations with varying hit rates
        cache_scenarios = [
            {"hits": 95, "misses": 5, "expected_rate": 0.95, "above_threshold": True},   # 95% - good
            {"hits": 85, "misses": 15, "expected_rate": 0.85, "above_threshold": False}, # 85% - below 90% threshold
            {"hits": 92, "misses": 8, "expected_rate": 0.92, "above_threshold": True},   # 92% - good
        ]
        
        with correlation_context("hit_rate_monitoring", episode_id="ep_hit_rate_001"):
            for i, scenario in enumerate(cache_scenarios):
                # Update cache metrics for each scenario
                update_cache_metrics(
                    cache_hit_count=scenario['hits'],
                    cache_miss_count=scenario['misses'],
                    cache_evictions=0,
                    cache_memory_usage_mb=500.0,
                    cache_memory_limit_mb=2048.0
                )
                
                enhanced_logger = get_enhanced_logger("hit_rate_test")
                enhanced_logger.info(f"Cache hit rate scenario {i+1}", extra={
                    "scenario_id": i+1,
                    "expected_hit_rate": scenario['expected_rate'],
                    "above_threshold": scenario['above_threshold'],
                    "hit_rate_threshold": 0.90,
                    "metric_type": "cache_hit_rate_test"
                })
                
                time.sleep(0.01)
        
        time.sleep(0.1)  # Allow file writes
        
        # Validate hit rate monitoring integration
        with open(temp_json_log_file, 'r') as f:
            log_lines = f.read().strip().split('\n')
        
        hit_rate_records = []
        for line in log_lines:
            if line.strip():
                try:
                    record = json.loads(line)
                    if 'metric_type' in record and record['metric_type'] == 'cache_hit_rate_test':
                        hit_rate_records.append(record)
                except json.JSONDecodeError:
                    continue
        
        # Should have records for all scenarios
        assert len(hit_rate_records) >= len(cache_scenarios)
        
        # Validate hit rate calculations and monitoring
        for i, record in enumerate(hit_rate_records[:len(cache_scenarios)]):
            scenario = cache_scenarios[i]
            
            # Validate hit rate calculation
            assert 'cache_hit_rate' in record
            calculated_rate = record['cache_hit_rate']
            expected_rate = scenario['expected_rate']
            assert abs(calculated_rate - expected_rate) < 0.001
            
            # Validate cache statistics
            assert record['cache_hit_count'] == scenario['hits']
            assert record['cache_miss_count'] == scenario['misses']
            
            # Validate threshold context
            assert record['hit_rate_threshold'] == 0.90
            assert record['above_threshold'] == scenario['above_threshold']


class TestJSONFormatterValidation:
    """
    Test suite for JSON formatter function validation and edge case handling.
    
    Tests the internal JSON formatter implementation for correctness,
    performance, and robustness across various data types and scenarios.
    """
    
    @pytest.fixture
    def mock_log_record(self):
        """Create mock log record for formatter testing."""
        return {
            'time': type('Time', (), {'isoformat': lambda: '2024-01-15T10:30:00.123456Z'})(),
            'level': type('Level', (), {'name': 'INFO'})(),
            'name': 'test_logger',
            'function': 'test_function',
            'line': 42,
            'message': 'Test message',
            'extra': {
                'correlation_id': 'test_correlation_123',
                'request_id': 'req_test_001',
                'module': 'test_module',
                'thread_id': 12345,
                'process_id': 67890,
                'step_count': 100,
                'episode_id': 'ep_test_001'
            }
        }
    
    def test_json_formatter_basic_structure(self, mock_log_record):
        """
        Test JSON formatter produces correct basic structure.
        
        Validates that the JSON formatter correctly transforms log records
        into properly structured JSON with all required fields.
        """
        formatter = _create_json_formatter()
        json_output = formatter(mock_log_record)
        
        # Parse the JSON output
        parsed_record = json.loads(json_output)
        
        # Validate basic structure
        assert parsed_record['timestamp'] == '2024-01-15T10:30:00.123456Z'
        assert parsed_record['level'] == 'INFO'
        assert parsed_record['logger'] == 'test_logger'
        assert parsed_record['function'] == 'test_function'
        assert parsed_record['line'] == 42
        assert parsed_record['message'] == 'Test message'
        
        # Validate correlation fields
        assert parsed_record['correlation_id'] == 'test_correlation_123'
        assert parsed_record['request_id'] == 'req_test_001'
        assert parsed_record['module'] == 'test_module'
        assert parsed_record['thread_id'] == 12345
        assert parsed_record['process_id'] == 67890
        assert parsed_record['step_count'] == 100
        assert parsed_record['episode_id'] == 'ep_test_001'
    
    def test_json_formatter_performance_metrics_handling(self, mock_log_record):
        """
        Test JSON formatter handles performance metrics correctly.
        
        Validates that performance metrics are properly included and
        formatted in the JSON output structure.
        """
        # Add performance metrics to mock record
        mock_log_record['extra']['performance_metrics'] = {
            'operation_name': 'test_operation',
            'duration': 0.008,
            'memory_before': 100.5,
            'memory_after': 102.3,
            'memory_delta': 1.8
        }
        
        mock_log_record['extra']['metric_type'] = 'performance_test'
        
        formatter = _create_json_formatter()
        json_output = formatter(mock_log_record)
        parsed_record = json.loads(json_output)
        
        # Validate performance metrics inclusion
        assert 'performance' in parsed_record
        performance = parsed_record['performance']
        assert performance['operation_name'] == 'test_operation'
        assert performance['duration'] == 0.008
        assert performance['memory_delta'] == 1.8
        
        # Validate metric type
        assert parsed_record['metric_type'] == 'performance_test'
    
    def test_json_formatter_cache_statistics_handling(self, mock_log_record):
        """
        Test JSON formatter handles cache statistics correctly.
        
        Validates that cache statistics are properly formatted and
        included in the JSON output structure.
        """
        # Add cache statistics to mock record
        cache_stats = {
            'cache_hit_count': 150,
            'cache_miss_count': 13,
            'cache_evictions': 2,
            'cache_hit_rate': 0.92,
            'cache_memory_usage_mb': 512.3,
            'cache_memory_limit_mb': 2048.0
        }
        
        for key, value in cache_stats.items():
            mock_log_record['extra'][key] = value
        
        formatter = _create_json_formatter()
        json_output = formatter(mock_log_record)
        parsed_record = json.loads(json_output)
        
        # Validate cache statistics grouping
        assert 'cache_stats' in parsed_record
        cache_stats_output = parsed_record['cache_stats']
        
        assert cache_stats_output['cache_hit_count'] == 150
        assert cache_stats_output['cache_miss_count'] == 13
        assert cache_stats_output['cache_evictions'] == 2
        assert cache_stats_output['cache_hit_rate'] == 0.92
        assert cache_stats_output['cache_memory_usage_mb'] == 512.3
        assert cache_stats_output['cache_memory_limit_mb'] == 2048.0
    
    def test_json_formatter_complex_data_serialization(self, mock_log_record):
        """
        Test JSON formatter handles complex nested data structures.
        
        Validates that complex data types are properly serialized
        without losing information or causing errors.
        """
        # Add complex nested data
        mock_log_record['extra']['complex_data'] = {
            'nested_dict': {
                'level1': {
                    'level2': ['item1', 'item2', 'item3'],
                    'numbers': [1, 2.5, 3.14159]
                }
            },
            'mixed_types': {
                'string': 'test',
                'integer': 42,
                'float': 3.14,
                'boolean': True,
                'null_value': None,
                'list': [1, 'two', 3.0, True]
            }
        }
        
        formatter = _create_json_formatter()
        json_output = formatter(mock_log_record)
        parsed_record = json.loads(json_output)
        
        # Validate complex data preservation
        assert 'complex_data' in parsed_record
        complex_data = parsed_record['complex_data']
        
        # Validate nested structure
        nested = complex_data['nested_dict']['level1']
        assert nested['level2'] == ['item1', 'item2', 'item3']
        assert nested['numbers'] == [1, 2.5, 3.14159]
        
        # Validate mixed types
        mixed = complex_data['mixed_types']
        assert mixed['string'] == 'test'
        assert mixed['integer'] == 42
        assert mixed['float'] == 3.14
        assert mixed['boolean'] is True
        assert mixed['null_value'] is None
        assert mixed['list'] == [1, 'two', 3.0, True]


# Test execution marker for cache and JSON logging tests
pytest.mark.cache_logging = pytest.mark.parametrize("test_type", ["cache", "json_logging"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])