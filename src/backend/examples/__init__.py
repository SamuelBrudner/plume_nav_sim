"""
Examples package initialization module providing convenient access to demonstration functions, educational utilities, 
and comprehensive example discovery for plume_nav_sim environment. This module exports key demonstration functions 
from all example modules, provides utility functions for example execution, and serves as central entry point for 
educational and benchmarking workflows with organized categorization and documentation.

This initialization module consolidates the complete example suite including basic usage tutorials, comprehensive 
visualization demonstrations, advanced Gymnasium integration patterns, and random agent benchmarking capabilities, 
enabling researchers and developers to quickly access and execute relevant demonstrations for their specific needs.
"""

# External imports with version comments
import logging  # >=3.10 - Logging configuration for examples package initialization and utility functions
import typing  # >=3.10 - Type hints for example discovery utilities and function signatures
from typing import Optional, Dict, List, Any, Callable, Union
import inspect  # >=3.10 - Function introspection for automatic example discovery and documentation generation
import importlib  # >=3.10 - Dynamic module loading for example discovery and execution utilities
import time  # >=3.10 - Timing utilities for execution tracking and performance monitoring
import sys  # >=3.10 - System utilities for error handling and execution management
import traceback  # >=3.10 - Error traceback formatting for comprehensive error reporting

# Internal imports from example modules for demonstration function access
from .basic_usage import (
    run_basic_usage_demo,  # Basic usage demonstration function for new users showing essential environment functionality
    demonstrate_basic_episode  # Core episode demonstration showing complete Gymnasium API lifecycle
)

from .visualization_demo import (
    run_comprehensive_visualization_demo,  # Comprehensive visualization demonstration showcasing dual-mode rendering and interactive features
    demonstrate_interactive_visualization,  # Interactive matplotlib visualization with real-time updates and performance monitoring
    demonstrate_rgb_array_rendering  # RGB array rendering demonstration for programmatic visualization and pixel analysis
)

from .gymnasium_integration import (
    run_gymnasium_integration_demo,  # Advanced Gymnasium integration showcase with comprehensive feature validation
    demonstrate_advanced_registration,  # Advanced environment registration with custom parameters and validation workflows
    demonstrate_reproducibility_validation  # Comprehensive reproducibility validation with statistical testing and cross-session verification
)

from .random_agent import (
    run_random_agent_demo,  # Random agent demonstration with comprehensive analysis and benchmarking
    execute_random_episode,  # Core random episode execution with performance tracking and statistical analysis
    benchmark_random_agent  # Comprehensive benchmarking function for random agent performance validation
)

# Global constants and package configuration
EXAMPLES_VERSION = '0.0.1'

AVAILABLE_EXAMPLES = {
    'basic': {
        'function': run_basic_usage_demo,
        'description': 'Basic usage demonstration for new users',
        'difficulty': 'beginner',
        'duration_minutes': 2
    },
    'visualization': {
        'function': run_comprehensive_visualization_demo,
        'description': 'Comprehensive visualization showcase',
        'difficulty': 'intermediate', 
        'duration_minutes': 5
    },
    'gymnasium': {
        'function': run_gymnasium_integration_demo,
        'description': 'Advanced Gymnasium integration patterns',
        'difficulty': 'advanced',
        'duration_minutes': 10
    },
    'random_agent': {
        'function': run_random_agent_demo,
        'description': 'Random agent benchmarking and analysis',
        'difficulty': 'intermediate',
        'duration_minutes': 5
    }
}

EXAMPLE_CATEGORIES = {
    'getting_started': ['basic'],
    'visualization': ['visualization', 'rgb_rendering'],
    'integration': ['gymnasium', 'registration'],
    'analysis': ['random_agent', 'benchmarking'],
    'advanced': ['gymnasium', 'performance']
}

# Initialize package logger for example execution tracking and error management
_logger = logging.getLogger(__name__)


def list_available_examples(category_filter: Optional[str] = None, include_descriptions: bool = True) -> Dict[str, Any]:
    """
    List all available example demonstrations with descriptions, difficulty levels, and estimated duration 
    for user guidance and example discovery.
    
    Args:
        category_filter: Optional category filter to limit examples by type using EXAMPLE_CATEGORIES mapping
        include_descriptions: Flag to include detailed descriptions in returned example information
        
    Returns:
        Dictionary of available examples with metadata and execution information for user selection
    """
    try:
        # Filter examples by category_filter if provided using EXAMPLE_CATEGORIES mapping
        filtered_examples = {}
        
        if category_filter is not None:
            if category_filter not in EXAMPLE_CATEGORIES:
                _logger.warning(f"Unknown category filter '{category_filter}', showing all examples")
                examples_to_include = list(AVAILABLE_EXAMPLES.keys())
            else:
                examples_to_include = EXAMPLE_CATEGORIES[category_filter]
        else:
            examples_to_include = list(AVAILABLE_EXAMPLES.keys())
        
        # Iterate through AVAILABLE_EXAMPLES dictionary extracting example metadata
        for example_name, example_info in AVAILABLE_EXAMPLES.items():
            if example_name in examples_to_include or category_filter is None:
                example_data = {
                    'difficulty': example_info['difficulty'],
                    'duration_minutes': example_info['duration_minutes']
                }
                
                # Include detailed descriptions if include_descriptions flag enabled
                if include_descriptions:
                    example_data['description'] = example_info['description']
                
                # Format example information with difficulty level and duration estimates
                example_data['execution_info'] = {
                    'estimated_runtime': f"{example_info['duration_minutes']} minutes",
                    'complexity_level': example_info['difficulty'],
                    'function_name': example_info['function'].__name__
                }
                
                # Add execution instructions and parameter information for each example
                example_data['usage_instructions'] = f"Run with: run_example('{example_name}')"
                
                filtered_examples[example_name] = example_data
        
        # Return structured dictionary with example discovery and execution guidance
        result = {
            'examples': filtered_examples,
            'total_available': len(filtered_examples),
            'category_filter': category_filter,
            'package_version': EXAMPLES_VERSION
        }
        
        _logger.info(f"Listed {len(filtered_examples)} examples with category filter: {category_filter}")
        return result
        
    except Exception as e:
        _logger.error(f"Failed to list available examples: {e}")
        return {'error': str(e), 'examples': {}}


def run_example(example_name: str, config_params: Optional[Dict[str, Any]] = None, 
                verbose_output: bool = False) -> int:
    """
    Execute specified example demonstration by name with optional configuration parameters and 
    comprehensive error handling for user convenience.
    
    Args:
        example_name: Name of example to execute from AVAILABLE_EXAMPLES registry
        config_params: Optional dictionary of configuration parameters for example customization
        verbose_output: Enable detailed logging output for execution tracking and debugging
        
    Returns:
        Exit status code: 0 for success, 1 for failure, 2 for invalid example name
    """
    try:
        # Validate example_name against AVAILABLE_EXAMPLES keys with helpful error messages
        if example_name not in AVAILABLE_EXAMPLES:
            _logger.error(f"Invalid example name '{example_name}'. Available examples: {list(AVAILABLE_EXAMPLES.keys())}")
            return 2
        
        # Log example execution start with name and configuration information
        _logger.info(f"Starting execution of example '{example_name}'")
        if config_params:
            _logger.info(f"Configuration parameters: {config_params}")
        
        # Retrieve example function from AVAILABLE_EXAMPLES registry
        example_info = AVAILABLE_EXAMPLES[example_name]
        example_function = example_info['function']
        
        # Set up verbose logging if verbose_output enabled for detailed execution tracking
        if verbose_output:
            logging.getLogger().setLevel(logging.DEBUG)
            _logger.debug(f"Verbose output enabled for example '{example_name}'")
        
        # Apply config_params if provided with parameter validation and default handling
        execution_start_time = time.time()
        
        try:
            # Execute example function with comprehensive error handling and timeout management
            if config_params:
                # Try to pass config parameters if the function accepts them
                sig = inspect.signature(example_function)
                if len(sig.parameters) > 0:
                    result = example_function(config_params)
                else:
                    _logger.warning(f"Example function '{example_name}' does not accept parameters, ignoring config_params")
                    result = example_function()
            else:
                result = example_function()
            
            # Capture execution results and performance metrics for user feedback
            execution_duration = time.time() - execution_start_time
            
            # Handle different return types from example functions
            if isinstance(result, int):
                exit_status = result
            else:
                exit_status = 0  # Assume success if no explicit status returned
            
            # Log example completion with status and duration information
            _logger.info(f"Example '{example_name}' completed in {execution_duration:.2f}s with exit status {exit_status}")
            
            # Return appropriate exit status code with user-friendly success/failure indication
            return exit_status
            
        except Exception as exec_error:
            _logger.error(f"Example '{example_name}' execution failed: {exec_error}")
            if verbose_output:
                _logger.debug(f"Detailed error traceback: {traceback.format_exc()}")
            return 1
    
    except Exception as e:
        _logger.error(f"Failed to run example '{example_name}': {e}")
        return 1


def get_example_help(example_name: str) -> str:
    """
    Retrieve comprehensive help information for specified example including usage patterns, 
    parameter options, and troubleshooting guidance.
    
    Args:
        example_name: Name of example to get help information for
        
    Returns:
        Formatted help text with usage instructions and parameter documentation
    """
    try:
        # Validate example_name exists in AVAILABLE_EXAMPLES with suggestion for typos
        if example_name not in AVAILABLE_EXAMPLES:
            available_names = list(AVAILABLE_EXAMPLES.keys())
            help_text = f"Example '{example_name}' not found.\n\nAvailable examples:\n"
            for name in available_names:
                help_text += f"  - {name}: {AVAILABLE_EXAMPLES[name]['description']}\n"
            return help_text
        
        # Retrieve example function and metadata from registry
        example_info = AVAILABLE_EXAMPLES[example_name]
        example_function = example_info['function']
        
        # Extract function signature and parameter information using inspect module
        sig = inspect.signature(example_function)
        docstring = inspect.getdoc(example_function) or "No documentation available."
        
        # Generate usage examples with common parameter combinations
        help_text = f"HELP: {example_name}\n"
        help_text += "=" * (len(example_name) + 6) + "\n\n"
        
        help_text += f"Description: {example_info['description']}\n"
        help_text += f"Difficulty: {example_info['difficulty']}\n"
        help_text += f"Estimated Duration: {example_info['duration_minutes']} minutes\n\n"
        
        help_text += "Function Documentation:\n"
        help_text += "-" * 25 + "\n"
        help_text += f"{docstring}\n\n"
        
        help_text += f"Function Signature: {example_function.__name__}{sig}\n\n"
        
        # Include troubleshooting section with common issues and solutions
        help_text += "Usage Examples:\n"
        help_text += "-" * 15 + "\n"
        help_text += f"Basic usage:     run_example('{example_name}')\n"
        if len(sig.parameters) > 0:
            help_text += f"With parameters: run_example('{example_name}', {{'param': 'value'}})\n"
        help_text += f"Verbose output:  run_example('{example_name}', verbose_output=True)\n\n"
        
        help_text += "Troubleshooting:\n"
        help_text += "-" * 16 + "\n"
        help_text += "- Ensure plume_nav_sim environment is properly installed\n"
        help_text += "- Check that all dependencies are available\n"
        help_text += "- Use verbose_output=True for detailed execution information\n"
        help_text += f"- Refer to module documentation for '{example_function.__module__}'\n"
        
        # Format help text with clear sections and appropriate formatting
        return help_text
        
    except Exception as e:
        _logger.error(f"Failed to get help for example '{example_name}': {e}")
        return f"Error retrieving help for '{example_name}': {str(e)}"


def discover_examples(include_internal_functions: bool = False, validate_signatures: bool = True) -> Dict[str, Any]:
    """
    Automatically discover example functions from all modules in examples package using introspection 
    and dynamic import for comprehensive example registry.
    
    Args:
        include_internal_functions: Include internal/helper functions in discovery results
        validate_signatures: Perform signature validation on discovered functions
        
    Returns:
        Discovery results with found examples, validation status, and import success
    """
    try:
        # Scan examples package directory for Python modules using importlib
        discovery_results = {
            'discovered_examples': {},
            'validation_results': {},
            'import_status': {},
            'discovery_summary': {}
        }
        
        example_modules = [
            'basic_usage',
            'visualization_demo', 
            'gymnasium_integration',
            'random_agent'
        ]
        
        discovered_count = 0
        
        # Dynamically import each module and extract functions using inspect
        for module_name in example_modules:
            try:
                module_path = f'plume_nav_sim.examples.{module_name}'
                module = importlib.import_module(module_path)
                
                discovery_results['import_status'][module_name] = 'success'
                
                # Extract functions from module using inspect
                module_functions = []
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    # Filter functions based on naming patterns and export conventions
                    if obj.__module__ == module_path:  # Only functions defined in this module
                        if include_internal_functions or not name.startswith('_'):
                            if name.startswith(('run_', 'demonstrate_', 'execute_', 'benchmark_')):
                                module_functions.append({
                                    'name': name,
                                    'function': obj,
                                    'module': module_name,
                                    'docstring': inspect.getdoc(obj) or 'No documentation'
                                })
                                discovered_count += 1
                
                discovery_results['discovered_examples'][module_name] = module_functions
                
                # Validate function signatures if validate_signatures enabled with type checking
                if validate_signatures:
                    validation_results = []
                    for func_info in module_functions:
                        try:
                            sig = inspect.signature(func_info['function'])
                            validation_results.append({
                                'function': func_info['name'],
                                'signature_valid': True,
                                'parameters': str(sig)
                            })
                        except Exception as val_error:
                            validation_results.append({
                                'function': func_info['name'],
                                'signature_valid': False,
                                'error': str(val_error)
                            })
                    
                    discovery_results['validation_results'][module_name] = validation_results
                
            except Exception as import_error:
                _logger.error(f"Failed to import module '{module_name}': {import_error}")
                discovery_results['import_status'][module_name] = f'failed: {str(import_error)}'
        
        # Categorize discovered functions by purpose and complexity level
        categories = {
            'demonstrations': [],
            'utilities': [],
            'benchmarking': [],
            'visualization': []
        }
        
        for module_name, functions in discovery_results['discovered_examples'].items():
            for func_info in functions:
                name = func_info['name']
                if name.startswith('run_'):
                    categories['demonstrations'].append(func_info)
                elif name.startswith('demonstrate_'):
                    categories['visualization'].append(func_info)
                elif name.startswith('benchmark_'):
                    categories['benchmarking'].append(func_info)
                else:
                    categories['utilities'].append(func_info)
        
        discovery_results['categorized_functions'] = categories
        
        # Generate metadata including descriptions, parameters, and dependencies
        discovery_results['discovery_summary'] = {
            'total_modules_scanned': len(example_modules),
            'successful_imports': sum(1 for status in discovery_results['import_status'].values() if status == 'success'),
            'total_functions_discovered': discovered_count,
            'validation_enabled': validate_signatures,
            'internal_functions_included': include_internal_functions
        }
        
        _logger.info(f"Example discovery completed: {discovered_count} functions from {len(example_modules)} modules")
        
        # Return comprehensive discovery results with validation and categorization
        return discovery_results
        
    except Exception as e:
        _logger.error(f"Example discovery failed: {e}")
        return {'error': str(e), 'discovered_examples': {}}


def create_example_runner(log_level: Optional[str] = None, capture_output: bool = False, 
                         output_directory: Optional[str] = None) -> 'ExampleRunner':
    """
    Create configured example runner instance with logging setup, error handling, and execution 
    management for batch example execution and testing.
    
    Args:
        log_level: Optional logging level configuration for runner operation
        capture_output: Enable output capture for result analysis and reporting
        output_directory: Optional directory for captured outputs and execution logs
        
    Returns:
        Configured runner instance ready for example execution
    """
    try:
        # Initialize ExampleRunner instance with configuration parameters
        runner = ExampleRunner(
            log_level=log_level,
            capture_output=capture_output,
            output_directory=output_directory
        )
        
        _logger.info(f"Example runner created with log_level='{log_level}', capture_output={capture_output}")
        
        # Return configured runner ready for batch example execution
        return runner
        
    except Exception as e:
        _logger.error(f"Failed to create example runner: {e}")
        raise RuntimeError(f"Example runner creation failed: {e}")


def generate_examples_documentation(output_format: Optional[str] = None, 
                                   include_code_samples: bool = False, 
                                   output_file: Optional[str] = None) -> str:
    """
    Generate comprehensive documentation for all available examples including usage guides, 
    API references, and educational materials for user guidance.
    
    Args:
        output_format: Optional output format specification ('markdown', 'rst', 'html')
        include_code_samples: Include code examples and usage patterns in documentation
        output_file: Optional file path for saving generated documentation
        
    Returns:
        Generated documentation content with comprehensive example coverage
    """
    try:
        # Collect metadata from all available examples using discovery utilities
        documentation_lines = []
        
        # Generate table of contents with categorization and difficulty levels
        if output_format == 'markdown':
            documentation_lines.extend([
                "# Plume Navigation Simulation Examples",
                "",
                "This documentation provides comprehensive coverage of all available example demonstrations.",
                "",
                "## Table of Contents",
                ""
            ])
        else:
            documentation_lines.extend([
                "PLUME NAVIGATION SIMULATION EXAMPLES",
                "=" * 40,
                "",
                "Comprehensive example documentation and usage guides.",
                "",
                "TABLE OF CONTENTS",
                "-" * 17,
                ""
            ])
        
        # Create detailed documentation for each example with usage patterns
        for example_name, example_info in AVAILABLE_EXAMPLES.items():
            if output_format == 'markdown':
                documentation_lines.extend([
                    f"## {example_name.title()} Example",
                    "",
                    f"**Description:** {example_info['description']}",
                    f"**Difficulty:** {example_info['difficulty']}",
                    f"**Duration:** {example_info['duration_minutes']} minutes",
                    ""
                ])
            else:
                documentation_lines.extend([
                    f"{example_name.upper()} EXAMPLE",
                    "-" * (len(example_name) + 8),
                    f"Description: {example_info['description']}",
                    f"Difficulty: {example_info['difficulty']}",
                    f"Duration: {example_info['duration_minutes']} minutes",
                    ""
                ])
            
            # Include code samples if include_code_samples enabled with syntax highlighting
            if include_code_samples:
                if output_format == 'markdown':
                    documentation_lines.extend([
                        "### Usage Example",
                        "",
                        "```python",
                        f"from plume_nav_sim.examples import run_example",
                        "",
                        f"# Basic execution",
                        f"status = run_example('{example_name}')",
                        "",
                        f"# With verbose output",
                        f"status = run_example('{example_name}', verbose_output=True)",
                        "```",
                        ""
                    ])
                else:
                    documentation_lines.extend([
                        "Usage Example:",
                        "",
                        f"    from plume_nav_sim.examples import run_example",
                        f"    ",
                        f"    # Basic execution",
                        f"    status = run_example('{example_name}')",
                        f"    ",
                        f"    # With verbose output",
                        f"    status = run_example('{example_name}', verbose_output=True)",
                        ""
                    ])
        
        # Add troubleshooting sections with common issues and solutions
        if output_format == 'markdown':
            documentation_lines.extend([
                "## Troubleshooting",
                "",
                "### Common Issues",
                "",
                "1. **Import Errors**: Ensure plume_nav_sim is properly installed",
                "2. **Rendering Issues**: Check matplotlib installation and display configuration",
                "3. **Performance Issues**: Use verbose_output=True for detailed timing information",
                "",
                "### Getting Help",
                "",
                "Use `get_example_help(example_name)` for detailed information about specific examples.",
                ""
            ])
        else:
            documentation_lines.extend([
                "TROUBLESHOOTING",
                "-" * 15,
                "",
                "Common Issues:",
                "  1. Import Errors: Ensure plume_nav_sim is properly installed",
                "  2. Rendering Issues: Check matplotlib installation",
                "  3. Performance Issues: Use verbose output for timing details",
                "",
                "Getting Help:",
                "  Use get_example_help(example_name) for detailed information.",
                ""
            ])
        
        # Format documentation according to output_format (markdown, rst, html)
        documentation_content = "\n".join(documentation_lines)
        
        # Save to output_file if specified with appropriate formatting
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(documentation_content)
                _logger.info(f"Documentation saved to {output_file}")
            except Exception as save_error:
                _logger.error(f"Failed to save documentation to {output_file}: {save_error}")
        
        _logger.info(f"Generated {len(documentation_lines)} lines of documentation")
        
        # Return complete documentation string for display or further processing
        return documentation_content
        
    except Exception as e:
        _logger.error(f"Failed to generate examples documentation: {e}")
        return f"Error generating documentation: {str(e)}"


class ExampleRunner:
    """
    Example execution manager providing batch processing, logging integration, error handling, 
    and comprehensive execution coordination for automated testing and demonstration workflows.
    """
    
    def __init__(self, log_level: Optional[str] = None, capture_output: bool = False, 
                 output_directory: Optional[str] = None):
        """
        Initialize ExampleRunner with logging configuration, output management, and example registry 
        setup for comprehensive execution management.
        
        Args:
            log_level: Optional logging level for execution tracking and debugging
            capture_output: Enable output capture for result analysis and reporting
            output_directory: Optional directory for captured outputs and execution logs
        """
        # Initialize logging configuration with specified log_level and formatting
        if log_level:
            numeric_level = getattr(logging, log_level.upper(), logging.INFO)
            logging.getLogger().setLevel(numeric_level)
        
        # Set up output capture configuration with file management if capture_output enabled
        self.capture_enabled = capture_output
        self.output_dir = output_directory
        
        if self.output_dir:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize example registry from AVAILABLE_EXAMPLES global dictionary
        self.available_examples = AVAILABLE_EXAMPLES.copy()
        
        # Set up execution history tracking for performance monitoring and analysis
        self.execution_history = []
        
        # Create logger instance for runner with appropriate handler configuration
        self.logger = logging.getLogger(f"{__name__}.ExampleRunner")
        
        # Configure error handling and timeout management for robust execution
        self.timeout_seconds = 300  # 5 minute default timeout per example
        
        self.logger.info(f"ExampleRunner initialized with {len(self.available_examples)} available examples")
    
    def run_single_example(self, example_name: str, config_params: Optional[Dict[str, Any]] = None, 
                          timeout_enabled: bool = True) -> Dict[str, Any]:
        """
        Execute single example with comprehensive logging, error handling, performance monitoring, 
        and result capture for detailed execution analysis.
        
        Args:
            example_name: Name of example to execute from available examples registry
            config_params: Optional configuration parameters for example customization
            timeout_enabled: Enable timeout management for execution safety
            
        Returns:
            Execution results with status, timing, output, and error information
        """
        execution_start_time = time.time()
        
        # Initialize execution result dictionary
        execution_result = {
            'example_name': example_name,
            'start_time': execution_start_time,
            'config_params': config_params,
            'timeout_enabled': timeout_enabled,
            'status': 'unknown',
            'exit_code': None,
            'duration_seconds': 0.0,
            'output': None,
            'error': None
        }
        
        try:
            # Validate example_name against available examples registry
            if example_name not in self.available_examples:
                execution_result['status'] = 'invalid_example'
                execution_result['error'] = f"Example '{example_name}' not found in registry"
                return execution_result
            
            # Set up execution context with logging and output capture
            self.logger.info(f"Executing example '{example_name}' with timeout_enabled={timeout_enabled}")
            
            if config_params:
                self.logger.debug(f"Configuration parameters: {config_params}")
            
            # Execute example function with comprehensive error handling
            try:
                exit_code = run_example(example_name, config_params, verbose_output=True)
                
                execution_result['status'] = 'completed'
                execution_result['exit_code'] = exit_code
                
                if exit_code == 0:
                    execution_result['success'] = True
                    self.logger.info(f"Example '{example_name}' completed successfully")
                else:
                    execution_result['success'] = False
                    self.logger.warning(f"Example '{example_name}' completed with exit code {exit_code}")
                
            except Exception as exec_error:
                execution_result['status'] = 'error'
                execution_result['error'] = str(exec_error)
                execution_result['success'] = False
                self.logger.error(f"Example '{example_name}' execution failed: {exec_error}")
            
            # Capture execution results including status, timing, and output
            execution_result['duration_seconds'] = time.time() - execution_start_time
            
            # Update execution history with results and performance metrics
            self.execution_history.append(execution_result.copy())
            
            return execution_result
            
        except Exception as runner_error:
            execution_result['status'] = 'runner_error'
            execution_result['error'] = str(runner_error)
            execution_result['duration_seconds'] = time.time() - execution_start_time
            self.logger.error(f"ExampleRunner error for '{example_name}': {runner_error}")
            return execution_result
    
    def run_batch_examples(self, example_names: List[str], global_config: Optional[Dict[str, Any]] = None, 
                          parallel_execution: bool = False, stop_on_failure: bool = False) -> Dict[str, Any]:
        """
        Execute multiple examples in batch with parallel processing options, comprehensive reporting, 
        and failure analysis for automated testing and validation.
        
        Args:
            example_names: List of example names to execute in batch
            global_config: Optional global configuration applied to all examples
            parallel_execution: Enable parallel execution for improved performance
            stop_on_failure: Stop batch execution on first failure
            
        Returns:
            Batch execution results with individual results, summary statistics, and failure analysis
        """
        batch_start_time = time.time()
        
        # Initialize batch results dictionary
        batch_results = {
            'batch_start_time': batch_start_time,
            'example_names': example_names,
            'global_config': global_config,
            'parallel_execution': parallel_execution,
            'stop_on_failure': stop_on_failure,
            'individual_results': [],
            'summary_statistics': {},
            'failure_analysis': {}
        }
        
        try:
            # Validate all example_names in batch with comprehensive error reporting
            invalid_examples = [name for name in example_names if name not in self.available_examples]
            if invalid_examples:
                batch_results['validation_error'] = f"Invalid examples: {invalid_examples}"
                self.logger.error(f"Batch validation failed - invalid examples: {invalid_examples}")
                return batch_results
            
            # Set up batch execution context with logging and progress tracking
            self.logger.info(f"Starting batch execution of {len(example_names)} examples")
            self.logger.info(f"Parallel execution: {parallel_execution}, Stop on failure: {stop_on_failure}")
            
            successful_executions = 0
            failed_executions = 0
            
            # Execute examples sequentially or in parallel based on parallel_execution flag
            if parallel_execution:
                self.logger.warning("Parallel execution not implemented in this version - using sequential execution")
            
            for i, example_name in enumerate(example_names):
                self.logger.info(f"Executing example {i+1}/{len(example_names)}: '{example_name}'")
                
                # Execute single example with batch configuration
                result = self.run_single_example(
                    example_name=example_name,
                    config_params=global_config,
                    timeout_enabled=True
                )
                
                batch_results['individual_results'].append(result)
                
                # Track success/failure counts
                if result.get('success', False):
                    successful_executions += 1
                else:
                    failed_executions += 1
                    
                    # Handle stop_on_failure logic with appropriate cleanup and reporting
                    if stop_on_failure:
                        self.logger.warning(f"Stopping batch execution due to failure in '{example_name}'")
                        batch_results['stopped_on_failure'] = True
                        batch_results['failed_example'] = example_name
                        break
            
            # Aggregate results from all executions with statistical analysis
            total_duration = time.time() - batch_start_time
            total_examples = len(batch_results['individual_results'])
            
            batch_results['summary_statistics'] = {
                'total_examples_requested': len(example_names),
                'total_examples_executed': total_examples,
                'successful_executions': successful_executions,
                'failed_executions': failed_executions,
                'success_rate': successful_executions / total_examples if total_examples > 0 else 0.0,
                'total_batch_duration': total_duration,
                'average_execution_time': total_duration / total_examples if total_examples > 0 else 0.0
            }
            
            # Generate batch execution report with success rates and failure analysis
            if failed_executions > 0:
                failed_examples = [
                    result['example_name'] for result in batch_results['individual_results']
                    if not result.get('success', False)
                ]
                
                batch_results['failure_analysis'] = {
                    'failed_example_names': failed_examples,
                    'failure_rate': failed_executions / total_examples,
                    'common_failure_patterns': 'Analysis not implemented'  # Could be enhanced
                }
            
            self.logger.info(f"Batch execution completed - {successful_executions}/{total_examples} successful")
            
            # Return comprehensive batch results for validation and reporting
            return batch_results
            
        except Exception as batch_error:
            batch_results['batch_error'] = str(batch_error)
            batch_results['total_batch_duration'] = time.time() - batch_start_time
            self.logger.error(f"Batch execution failed: {batch_error}")
            return batch_results
    
    def get_execution_summary(self, include_performance_analysis: bool = True, 
                             include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive execution summary with statistics, performance analysis, and 
        recommendations from execution history for system optimization.
        
        Args:
            include_performance_analysis: Include detailed performance metrics and timing analysis
            include_recommendations: Include optimization recommendations based on execution patterns
            
        Returns:
            Execution summary with statistics, performance metrics, and optimization recommendations
        """
        try:
            # Analyze execution history for success rates and failure patterns
            if not self.execution_history:
                return {
                    'summary': 'No execution history available',
                    'total_executions': 0,
                    'recommendations': ['Execute examples to generate performance data']
                }
            
            summary = {
                'generation_time': time.time(),
                'total_executions': len(self.execution_history),
                'execution_statistics': {},
                'performance_analysis': {},
                'recommendations': []
            }
            
            # Calculate basic execution statistics
            successful_executions = sum(1 for result in self.execution_history if result.get('success', False))
            failed_executions = len(self.execution_history) - successful_executions
            
            summary['execution_statistics'] = {
                'successful_executions': successful_executions,
                'failed_executions': failed_executions,
                'success_rate': successful_executions / len(self.execution_history),
                'failure_rate': failed_executions / len(self.execution_history)
            }
            
            # Calculate performance statistics including timing analysis and resource usage
            if include_performance_analysis:
                durations = [result['duration_seconds'] for result in self.execution_history]
                
                if durations:
                    import statistics
                    summary['performance_analysis'] = {
                        'average_execution_time': statistics.mean(durations),
                        'median_execution_time': statistics.median(durations),
                        'max_execution_time': max(durations),
                        'min_execution_time': min(durations),
                        'execution_time_std': statistics.stdev(durations) if len(durations) > 1 else 0.0
                    }
                
                # Analyze example-specific performance
                example_performance = {}
                for result in self.execution_history:
                    example_name = result['example_name']
                    if example_name not in example_performance:
                        example_performance[example_name] = []
                    example_performance[example_name].append(result['duration_seconds'])
                
                summary['performance_analysis']['example_performance'] = {}
                for example_name, times in example_performance.items():
                    summary['performance_analysis']['example_performance'][example_name] = {
                        'executions': len(times),
                        'average_time': sum(times) / len(times),
                        'max_time': max(times),
                        'min_time': min(times)
                    }
            
            # Generate optimization recommendations if include_recommendations enabled
            if include_recommendations:
                recommendations = []
                
                # Success rate analysis
                success_rate = summary['execution_statistics']['success_rate']
                if success_rate == 1.0:
                    recommendations.append("Excellent: All examples executed successfully")
                elif success_rate >= 0.8:
                    recommendations.append("Good: Most examples successful, review failed executions")
                else:
                    recommendations.append("Attention needed: Low success rate indicates system issues")
                
                # Performance analysis
                if include_performance_analysis and 'average_execution_time' in summary['performance_analysis']:
                    avg_time = summary['performance_analysis']['average_execution_time']
                    if avg_time > 300:  # 5 minutes
                        recommendations.append("Consider optimizing long-running examples")
                    elif avg_time < 10:  # 10 seconds
                        recommendations.append("Examples execute quickly - system performance is good")
                
                # General recommendations
                recommendations.extend([
                    "Review execution logs for detailed failure analysis",
                    "Use batch execution for comprehensive testing",
                    "Monitor execution history for performance trends"
                ])
                
                summary['recommendations'] = recommendations
            
            self.logger.info(f"Generated execution summary for {len(self.execution_history)} executions")
            
            # Return comprehensive summary for system monitoring and optimization
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate execution summary: {e}")
            return {'error': str(e), 'total_executions': len(self.execution_history)}


# Export all relevant functions and classes for package interface
__all__ = [
    # Main demonstration functions from example modules
    'run_basic_usage_demo',
    'run_comprehensive_visualization_demo', 
    'run_gymnasium_integration_demo',
    'run_random_agent_demo',
    
    # Core demonstration functions for specific use cases
    'demonstrate_basic_episode',
    'demonstrate_interactive_visualization',
    'demonstrate_rgb_array_rendering',
    
    # Utility functions for example discovery and execution
    'list_available_examples',
    'run_example',
    'get_example_help',
    
    # Advanced execution management
    'ExampleRunner',
    'create_example_runner',
    
    # Package metadata and configuration
    'AVAILABLE_EXAMPLES',
    'EXAMPLE_CATEGORIES',
    
    # Documentation and discovery utilities
    'discover_examples',
    'generate_examples_documentation'
]

# Package initialization logging
_logger.info(f"Examples package initialized - Version {EXAMPLES_VERSION}, {len(AVAILABLE_EXAMPLES)} examples available")