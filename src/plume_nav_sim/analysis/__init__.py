"""
Analysis module providing the public API facade for automated research metrics calculation.

This module serves as the primary entry point for the plume navigation analysis subsystem,
exposing StatsAggregator implementations, factory functions, validation utilities, and 
protocol interfaces for comprehensive research metrics calculation and standardized 
summary generation. The module is designed to integrate seamlessly with the recorder 
system and Hydra configuration management while achieving ≤33 ms/step performance 
with 100 agents through optimized statistical processing.

Key Features:
    - Unified API for automated statistics collection via StatsAggregatorProtocol interface
    - Configurable metrics definitions with custom calculation support per Section 5.2.9
    - Integration with Recorder system for data source connectivity and validation
    - Export functionality for summary.json with standardized research metrics format
    - Performance optimization achieving ≤33 ms/step with 100 agents through efficient algorithms
    - Hydra configuration integration for component discovery and runtime selection
    - Validation utilities for statistics configuration compatibility with recorder backends

Architecture Integration:
    This module implements the analysis components required by Section 0.2.1 of the
    technical specification, providing the statistics aggregation layer that enables
    automated research metrics collection. It serves as the bridge between simulation
    data (via the recorder system) and research analysis workflows, with standardized
    output formats for cross-project comparison and reproducibility.

Performance Requirements:
    - Statistics calculation: ≤33 ms/step with 100 agents per specification requirement
    - Memory-efficient algorithms for large dataset processing with configurable limits
    - Parallel processing support for batch statistics calculation when enabled
    - Recorder integration with <1ms additional overhead for data connectivity
    - Configurable aggregation levels supporting episode, run, and batch analysis

Examples:
    Basic statistics aggregator with default configuration:
        >>> from plume_nav_sim.analysis import create_stats_aggregator
        >>> aggregator = create_stats_aggregator({
        ...     'metrics_definitions': {'trajectory': ['mean', 'std', 'efficiency']},
        ...     'aggregation_levels': ['episode', 'run'],
        ...     'performance_tracking': True
        ... })
        >>> episode_stats = aggregator.calculate_episode_stats(episode_data)
        >>> aggregator.export_summary('./results/summary.json')

    Integration with recorder system for automated data collection:
        >>> from plume_nav_sim.analysis import validate_recorder_compatibility
        >>> from plume_nav_sim.recording import RecorderFactory
        >>> recorder = RecorderFactory.create_recorder({'backend': 'parquet'})
        >>> compatibility = validate_recorder_compatibility(aggregator, recorder)
        >>> if compatibility['compatible']:
        ...     print("Statistics aggregator compatible with recorder backend")

    Hydra configuration-driven component creation:
        >>> from plume_nav_sim.analysis import create_from_config
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     aggregator = create_from_config(cfg.analysis.stats_aggregator)
        ...     summary = generate_summary(aggregator, episodes_data)

    Advanced statistics with custom calculation functions:
        >>> def custom_tortuosity(trajectory_data):
        ...     path_length = calculate_path_length(trajectory_data)
        ...     direct_distance = calculate_direct_distance(trajectory_data)
        ...     return path_length / direct_distance if direct_distance > 0 else float('inf')
        >>> 
        >>> config = StatsAggregatorConfig(
        ...     custom_calculations={'tortuosity': custom_tortuosity},
        ...     performance_tracking=True,
        ...     parallel_processing=True
        ... )
        >>> aggregator = StatsAggregator(config)
        >>> metrics = aggregator.get_performance_metrics()
        >>> print(f"Processing efficiency: {metrics['episodes_per_second']:.1f} eps/s")
"""

from __future__ import annotations
import warnings
import time
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
import logging

# Core protocol imports for interface compliance
from ..core.protocols import StatsAggregatorProtocol

# Analysis implementation imports
from .stats import (
    StatsAggregator,
    StatsAggregatorConfig,
    calculate_basic_stats,
    calculate_advanced_stats,
    create_stats_aggregator,
    generate_summary_report
)

# Recording system integration
from ..recording import RecorderFactory

# Configure module logging
logger = logging.getLogger(__name__)

# Hydra configuration support (fail fast if unavailable)
try:
    from hydra import instantiate
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError as e:  # pragma: no cover - executed when Hydra is missing
    logger.error("Hydra is required for configuration-driven analysis: %s", e)
    raise

# Module version for API compatibility tracking
__version__ = "1.0.0"


class AnalysisModuleError(Exception):
    """Base exception class for analysis module errors."""
    pass


class ConfigurationError(AnalysisModuleError):
    """Exception raised for analysis configuration errors."""
    pass


class RecorderCompatibilityError(AnalysisModuleError):
    """Exception raised for recorder compatibility issues."""
    pass


def create_from_config(config: Union[Dict[str, Any], DictConfig]) -> StatsAggregatorProtocol:
    """
    Create statistics aggregator from Hydra configuration with validation and error handling.
    
    This function provides the primary configuration-driven entry point for creating
    statistics aggregators from Hydra configuration objects. It supports both direct
    configuration dictionaries and DictConfig objects with automatic parameter validation,
    dependency checking, and performance optimization recommendations.
    
    Args:
        config: Configuration object specifying aggregator parameters and settings.
            Supports both dictionary format and Hydra DictConfig with automatic
            conversion and validation. Required fields include metrics_definitions
            and aggregation_levels, with optional performance and integration settings.
            
    Returns:
        StatsAggregatorProtocol: Configured statistics aggregator instance implementing
            the full protocol interface with performance monitoring and validation.
            
    Raises:
        ConfigurationError: If configuration parameters are invalid or incompatible
        ImportError: If required dependencies for specified features are missing
        ValueError: If configuration values are outside valid ranges or types
        
    Examples:
        From Hydra DictConfig in research workflow:
        >>> config = DictConfig({
        ...     'metrics_definitions': {
        ...         'trajectory': ['mean', 'std', 'efficiency', 'tortuosity'],
        ...         'concentration': ['detection_rate', 'mean', 'percentiles']
        ...     },
        ...     'aggregation_levels': ['episode', 'run'],
        ...     'performance_tracking': True,
        ...     'parallel_processing': False,
        ...     'output_format': 'json'
        ... })
        >>> aggregator = create_from_config(config)
        
        With custom calculation functions:
        >>> def path_efficiency(trajectory):
        ...     return calculate_efficiency_metric(trajectory)
        >>> config = {
        ...     'metrics_definitions': {'trajectory': ['mean', 'custom']},
        ...     'custom_calculations': {'path_efficiency': path_efficiency},
        ...     'aggregation_levels': ['episode', 'run', 'batch']
        ... }
        >>> aggregator = create_from_config(config)
        
        High-performance configuration for large datasets:
        >>> config = {
        ...     'metrics_definitions': {'trajectory': ['basic_stats_only']},
        ...     'performance_tracking': True,
        ...     'parallel_processing': True,
        ...     'memory_limit_mb': 1024,
        ...     'precision_mode': 'float32'  # Faster processing
        ... }
        >>> aggregator = create_from_config(config)
    """
    try:
        # Validate Hydra availability for advanced features
        if isinstance(config, DictConfig) and not HYDRA_AVAILABLE:
            warnings.warn(
                "Hydra not available. Some configuration features may be limited. "
                "Install hydra-core for full functionality.",
                UserWarning,
                stacklevel=2
            )
        
        # Handle different configuration types
        if isinstance(config, DictConfig):
            # Check for Hydra instantiation target
            if '_target_' in config:
                logger.debug("Using Hydra instantiate for component creation")
                return instantiate(config)
            else:
                # Convert DictConfig to dict for StatsAggregatorConfig
                config_dict = {k: v for k, v in config.items() if not k.startswith('_')}
                stats_config = StatsAggregatorConfig(**config_dict)
        elif isinstance(config, dict):
            # Validate required fields
            if 'metrics_definitions' not in config:
                raise ConfigurationError("metrics_definitions is required in configuration")
            if 'aggregation_levels' not in config:
                config['aggregation_levels'] = ['episode', 'run']  # Provide default
                
            stats_config = StatsAggregatorConfig(**config)
        else:
            raise ConfigurationError(f"Invalid configuration type: {type(config)}")
        
        # Validate performance requirements
        if hasattr(stats_config, 'memory_limit_mb') and stats_config.memory_limit_mb < 64:
            warnings.warn(
                "Low memory limit may impact performance with large datasets. "
                "Consider increasing memory_limit_mb to at least 64MB.",
                UserWarning,
                stacklevel=2
            )
        
        if hasattr(stats_config, 'buffer_size') and stats_config.buffer_size < 100:
            logger.info("Small buffer size detected. Consider increasing for better I/O performance.")
        
        # Create aggregator instance
        aggregator = StatsAggregator(stats_config)
        
        # Validate performance characteristics
        performance_metrics = aggregator.get_performance_metrics()
        logger.debug(f"Created stats aggregator with config: {stats_config}")
        
        return aggregator
        
    except Exception as e:
        logger.error(f"Failed to create stats aggregator from config: {e}")
        if isinstance(e, (ConfigurationError, ImportError, ValueError)):
            raise
        else:
            raise ConfigurationError(f"Configuration processing failed: {e}") from e


def validate_recorder_compatibility(
    aggregator: StatsAggregatorProtocol,
    recorder: Any,
    strict_validation: bool = True
) -> Dict[str, Any]:
    """
    Validate compatibility between statistics aggregator and recorder backend.
    
    This function performs comprehensive compatibility checking between statistics
    aggregators and recorder backends to ensure seamless data flow and prevent
    runtime errors. It validates data format compatibility, performance characteristics,
    and configuration alignment while providing detailed diagnostics and recommendations.
    
    Args:
        aggregator: Statistics aggregator instance implementing StatsAggregatorProtocol
        recorder: Recorder instance (typically from RecorderFactory)
        strict_validation: Enable strict validation with detailed checks (default: True)
        
    Returns:
        Dict[str, Any]: Comprehensive compatibility results with diagnostics:
            - compatible: bool indicating overall compatibility
            - backend_supported: bool for backend-specific support
            - performance_compatible: bool for performance requirement alignment
            - warnings: List of compatibility warnings and recommendations
            - errors: List of critical compatibility errors
            - recommendations: List of optimization suggestions
            
    Raises:
        RecorderCompatibilityError: If critical compatibility issues are detected
        ValueError: If invalid aggregator or recorder instances are provided
        
    Examples:
        Basic compatibility checking:
        >>> aggregator = create_stats_aggregator({'aggregation_levels': ['episode']})
        >>> recorder = RecorderFactory.create_recorder({'backend': 'parquet'})
        >>> compatibility = validate_recorder_compatibility(aggregator, recorder)
        >>> if compatibility['compatible']:
        ...     print("Compatible: Ready for integrated analysis workflow")
        ... else:
        ...     print(f"Issues detected: {compatibility['errors']}")
        
        Performance-focused validation:
        >>> compatibility = validate_recorder_compatibility(
        ...     aggregator, recorder, strict_validation=True
        ... )
        >>> if not compatibility['performance_compatible']:
        ...     print("Performance recommendations:")
        ...     for rec in compatibility['recommendations']:
        ...         print(f"  - {rec}")
        
        Integration workflow validation:
        >>> for backend in ['parquet', 'hdf5', 'sqlite']:
        ...     test_recorder = RecorderFactory.create_recorder({'backend': backend})
        ...     compat = validate_recorder_compatibility(aggregator, test_recorder)
        ...     print(f"{backend}: {'✓' if compat['compatible'] else '✗'}")
    """
    try:
        validation_results = {
            'compatible': True,
            'backend_supported': True,
            'performance_compatible': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Validate aggregator protocol compliance
        if not isinstance(aggregator, StatsAggregatorProtocol):
            validation_results['errors'].append(
                "Aggregator does not implement StatsAggregatorProtocol"
            )
            validation_results['compatible'] = False
        
        # Validate recorder availability and type
        if recorder is None:
            validation_results['errors'].append("Recorder instance is None")
            validation_results['compatible'] = False
            return validation_results
        
        # Get recorder backend information
        recorder_config = getattr(recorder, 'config', None)
        if recorder_config:
            if hasattr(recorder_config, 'backend'):
                recorder_backend = recorder_config.backend
            elif isinstance(recorder_config, dict):
                recorder_backend = recorder_config.get('backend', 'unknown')
            else:
                recorder_backend = 'unknown'
        else:
            recorder_backend = 'unknown'
            
        if recorder_backend == 'unknown':
            validation_results['warnings'].append(
                "Unable to determine recorder backend type"
            )
        
        # Check backend-specific compatibility
        try:
            supported_backends = RecorderFactory.get_available_backends()
            if recorder_backend not in supported_backends:
                validation_results['backend_supported'] = False
                validation_results['errors'].append(
                    f"Recorder backend '{recorder_backend}' not in supported backends: {supported_backends}"
                )
        except (AttributeError, ImportError):
            # RecorderFactory may not have get_available_backends method or may not be importable
            validation_results['warnings'].append(
                "Unable to verify backend compatibility - RecorderFactory method unavailable"
            )
        
        # Performance compatibility validation
        if strict_validation:
            # Get aggregator performance characteristics
            if hasattr(aggregator, 'get_performance_metrics'):
                agg_metrics = aggregator.get_performance_metrics()
                agg_compliance = agg_metrics.get('performance_compliance', {})
                
                if not agg_compliance.get('meets_33ms_target', True):
                    validation_results['performance_compatible'] = False
                    validation_results['warnings'].append(
                        "Aggregator may not meet 33ms/step performance target"
                    )
                    validation_results['recommendations'].append(
                        "Consider reducing metrics_definitions complexity or enabling parallel_processing"
                    )
            
            # Get recorder performance characteristics
            if hasattr(recorder, 'get_performance_metrics'):
                rec_metrics = recorder.get_performance_metrics()
                avg_write_time = rec_metrics.get('average_write_time', 0)
                
                if avg_write_time > 0.030:  # 30ms threshold
                    validation_results['performance_compatible'] = False
                    validation_results['warnings'].append(
                        f"Recorder write time ({avg_write_time*1000:.1f}ms) may impact performance"
                    )
                    validation_results['recommendations'].append(
                        "Consider increasing recorder buffer_size or enabling async_io"
                    )
        
        # Data format compatibility
        if hasattr(aggregator, 'config'):
            agg_config = aggregator.config
            
            # Check memory constraints alignment
            if hasattr(recorder, 'config'):
                rec_config = recorder.config
                agg_memory = getattr(agg_config, 'memory_limit_mb', 256)
                rec_memory = getattr(rec_config, 'memory_limit_mb', 256)
                
                total_memory = agg_memory + rec_memory
                if total_memory > 1024:  # 1GB threshold
                    validation_results['warnings'].append(
                        f"Combined memory usage ({total_memory}MB) may be high for system resources"
                    )
                    validation_results['recommendations'].append(
                        "Consider reducing memory limits or using memory-efficient precision modes"
                    )
            
            # Check output format compatibility
            output_format = getattr(agg_config, 'output_format', 'json')
            if recorder_backend == 'sqlite' and output_format not in ['json', 'pickle']:
                validation_results['warnings'].append(
                    f"Output format '{output_format}' may not be optimal for SQLite backend"
                )
                validation_results['recommendations'].append(
                    "Consider using 'json' output format for SQLite backend compatibility"
                )
        
        # Aggregation level validation
        if hasattr(aggregator, 'get_aggregation_levels'):
            agg_levels = aggregator.get_aggregation_levels()
            if 'batch' in agg_levels and recorder_backend == 'none':
                validation_results['warnings'].append(
                    "Batch aggregation specified but recorder backend is 'none'"
                )
                validation_results['recommendations'].append(
                    "Enable a data-persistent recorder backend for batch aggregation"
                )
        
        # Overall compatibility assessment
        if validation_results['errors']:
            validation_results['compatible'] = False
        elif len(validation_results['warnings']) > 3:
            validation_results['compatible'] = False
            validation_results['errors'].append(
                "Too many compatibility warnings indicate potential integration issues"
            )
        
        # Add integration recommendations
        if validation_results['compatible']:
            validation_results['recommendations'].append(
                "Integration validated successfully - ready for production use"
            )
            if strict_validation and validation_results['performance_compatible']:
                validation_results['recommendations'].append(
                    "Performance characteristics meet requirements for real-time analysis"
                )
        
        logger.debug(f"Recorder compatibility validation completed: {validation_results['compatible']}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Error during recorder compatibility validation: {e}")
        raise RecorderCompatibilityError(f"Validation failed: {e}") from e


def check_analysis_dependencies() -> Dict[str, Any]:
    """
    Check availability of analysis dependencies with detailed reporting.
    
    This function performs comprehensive dependency checking for the analysis module,
    validating the availability of optional dependencies, their versions, and
    functionality. Provides detailed reporting for troubleshooting and optimization
    guidance for different deployment scenarios.
    
    Returns:
        Dict[str, Any]: Comprehensive dependency status report:
            - all_available: bool indicating if all core dependencies are available
            - core_dependencies: Dict mapping core dependency names to availability status
            - optional_dependencies: Dict mapping optional dependency names to status
            - version_info: Dict mapping package names to version strings
            - warnings: List of dependency warnings and compatibility issues
            - recommendations: List of installation and optimization recommendations
            
    Examples:
        Basic dependency checking for deployment validation:
        >>> deps = check_analysis_dependencies()
        >>> if deps['all_available']:
        ...     print("✓ All analysis dependencies available")
        ... else:
        ...     print("⚠ Missing dependencies:")
        ...     for name, available in deps['core_dependencies'].items():
        ...         if not available:
        ...             print(f"  - {name}")
        
        Detailed dependency analysis for troubleshooting:
        >>> deps = check_analysis_dependencies()
        >>> print("Dependency Report:")
        >>> for name, version in deps['version_info'].items():
        ...     status = "✓" if name in deps['core_dependencies'] else "○"
        ...     print(f"  {status} {name}: {version}")
        >>> 
        >>> if deps['warnings']:
        ...     print("Warnings:")
        ...     for warning in deps['warnings']:
        ...         print(f"  ⚠ {warning}")
        
        Installation recommendations:
        >>> deps = check_analysis_dependencies()
        >>> if deps['recommendations']:
        ...     print("Recommendations:")
        ...     for rec in deps['recommendations']:
        ...         print(f"  • {rec}")
    """
    dependency_status = {
        'all_available': True,
        'core_dependencies': {},
        'optional_dependencies': {},
        'version_info': {},
        'warnings': [],
        'recommendations': []
    }
    
    # Core dependencies (required for basic functionality)
    core_deps = {
        'numpy': 'numpy',
        'scipy': 'scipy.stats',
        'logging': 'logging'  # Standard library
    }
    
    # Optional dependencies (enhance functionality but not required)
    optional_deps = {
        'hydra': 'hydra',
        'omegaconf': 'omegaconf',
        'pandas': 'pandas',
        'pyarrow': 'pyarrow',
        'h5py': 'h5py',
        'yaml': 'yaml',
        'psutil': 'psutil'
    }
    
    # Check core dependencies
    for dep_name, import_name in core_deps.items():
        try:
            if dep_name == 'logging':
                # Standard library - always available
                dependency_status['core_dependencies'][dep_name] = True
                dependency_status['version_info'][dep_name] = 'standard_library'
            else:
                module = __import__(import_name)
                dependency_status['core_dependencies'][dep_name] = True
                
                # Get version information
                version = getattr(module, '__version__', 'unknown')
                dependency_status['version_info'][dep_name] = version
                
                # Version-specific warnings
                if dep_name == 'numpy':
                    if hasattr(module, '__version__'):
                        import re
                        version_match = re.match(r'(\d+)\.(\d+)', module.__version__)
                        if version_match:
                            major, minor = map(int, version_match.groups())
                            if (major, minor) < (1, 26):
                                dependency_status['warnings'].append(
                                    f"NumPy version {module.__version__} is old. "
                                    "Consider upgrading to >=1.26.0 for improved performance."
                                )
                                
        except ImportError:
            dependency_status['core_dependencies'][dep_name] = False
            dependency_status['all_available'] = False
            dependency_status['warnings'].append(f"Core dependency '{dep_name}' not available")
    
    # Check optional dependencies
    for dep_name, import_name in optional_deps.items():
        try:
            module = __import__(import_name)
            dependency_status['optional_dependencies'][dep_name] = True
            
            # Get version information
            version = getattr(module, '__version__', 'unknown')
            dependency_status['version_info'][dep_name] = version
            
            # Specific version recommendations
            if dep_name == 'hydra' and hasattr(module, '__version__'):
                import re
                version_match = re.match(r'(\d+)\.(\d+)', module.__version__)
                if version_match:
                    major, minor = map(int, version_match.groups())
                    if (major, minor) < (1, 3):
                        dependency_status['recommendations'].append(
                            f"Hydra {module.__version__} detected. "
                            "Upgrade to >=1.3.0 for enhanced configuration features."
                        )
                        
        except ImportError:
            dependency_status['optional_dependencies'][dep_name] = False
            
            # Add recommendations for missing optional dependencies
            if dep_name == 'hydra':
                dependency_status['recommendations'].append(
                    "Install hydra-core for configuration-driven component creation: "
                    "pip install hydra-core"
                )
            elif dep_name == 'pandas':
                dependency_status['recommendations'].append(
                    "Install pandas for enhanced data export capabilities: "
                    "pip install pandas"
                )
            elif dep_name == 'pyarrow':
                dependency_status['recommendations'].append(
                    "Install pyarrow for high-performance parquet support: "
                    "pip install pyarrow"
                )
    
    # Check Hydra availability status
    dependency_status['hydra_available'] = HYDRA_AVAILABLE
    if not HYDRA_AVAILABLE:
        dependency_status['warnings'].append(
            "Hydra configuration features disabled. Some advanced functionality unavailable."
        )
    
    # Performance recommendations based on available dependencies
    if dependency_status['optional_dependencies'].get('psutil', False):
        dependency_status['recommendations'].append(
            "psutil available - enhanced memory monitoring enabled"
        )
    else:
        dependency_status['recommendations'].append(
            "Install psutil for enhanced memory monitoring: pip install psutil"
        )
    
    # Recorder integration recommendations
    if not dependency_status['optional_dependencies'].get('pandas', False):
        dependency_status['warnings'].append(
            "Pandas not available - limited recorder backend compatibility"
        )
    
    if not dependency_status['optional_dependencies'].get('h5py', False):
        dependency_status['warnings'].append(
            "h5py not available - HDF5 recorder backend not supported"
        )
    
    # Overall status assessment
    optional_available = sum(dependency_status['optional_dependencies'].values())
    total_optional = len(optional_deps)
    
    if optional_available / total_optional >= 0.8:
        dependency_status['recommendations'].append(
            "Excellent dependency coverage - all advanced features available"
        )
    elif optional_available / total_optional >= 0.5:
        dependency_status['recommendations'].append(
            "Good dependency coverage - most features available"
        )
    else:
        dependency_status['warnings'].append(
            "Limited dependency coverage - consider installing optional packages for full functionality"
        )
    
    logger.debug(f"Analysis dependencies checked: {dependency_status['all_available']}")
    return dependency_status


def generate_summary(
    aggregator: StatsAggregatorProtocol,
    episodes_data: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    **export_options: Any
) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics from episodes data with export options.
    
    This function provides a high-level interface for generating comprehensive statistical
    summaries from episode data using the specified aggregator. It handles run-level
    aggregation, performance monitoring, and optional export to standardized formats
    suitable for research reproducibility and cross-project comparison.
    
    Args:
        aggregator: Statistics aggregator instance implementing StatsAggregatorProtocol
        episodes_data: List of episode data dictionaries for analysis
        output_path: Optional file path for summary export (enables automatic export)
        **export_options: Additional options for summary export (format, compression, etc.)
        
    Returns:
        Dict[str, Any]: Comprehensive summary statistics with metadata:
            - run_statistics: Aggregated run-level statistics from all episodes
            - performance_metrics: Processing performance and timing information
            - episode_count: Total number of episodes processed
            - processing_time: Total time spent on statistical calculations
            - validation_results: Data validation and quality assessment results
            - export_status: Export completion status if output_path specified
            
    Raises:
        ValueError: If episodes_data is empty or aggregator is invalid
        AnalysisModuleError: If summary generation fails due to processing errors
        IOError: If export fails due to file system or permission errors
        
    Examples:
        Basic summary generation for research analysis:
        >>> aggregator = create_stats_aggregator({
        ...     'metrics_definitions': {
        ...         'trajectory': ['mean', 'std', 'efficiency'],
        ...         'concentration': ['detection_rate', 'mean']
        ...     },
        ...     'aggregation_levels': ['episode', 'run']
        ... })
        >>> summary = generate_summary(aggregator, episodes_data)
        >>> print(f"Processed {summary['episode_count']} episodes")
        >>> print(f"Average detection rate: {summary['run_statistics']['concentration_aggregation']['detection_rate_mean']:.3f}")
        
        Summary with automatic export for reproducibility:
        >>> summary = generate_summary(
        ...     aggregator, 
        ...     episodes_data,
        ...     output_path='./results/experiment_summary.json',
        ...     format='json',
        ...     indent=2
        ... )
        >>> if summary['export_status']:
        ...     print("Summary exported successfully for long-term storage")
        
        Performance-monitored summary generation:
        >>> import time
        >>> start_time = time.time()
        >>> summary = generate_summary(aggregator, episodes_data)
        >>> processing_time = time.time() - start_time
        >>> efficiency = len(episodes_data) / processing_time
        >>> print(f"Processing efficiency: {efficiency:.1f} episodes/second")
        >>> 
        >>> # Compare with aggregator's internal performance metrics
        >>> agg_metrics = summary['performance_metrics']
        >>> print(f"Internal timing: {agg_metrics['computation_time_ms']:.1f}ms")
    """
    try:
        # Validate input parameters
        if not episodes_data:
            raise ValueError("episodes_data cannot be empty")
        
        if not isinstance(aggregator, StatsAggregatorProtocol):
            raise ValueError("aggregator must implement StatsAggregatorProtocol")
        
        # Record processing start time
        processing_start_time = time.time()
        
        # Validate episodes data structure
        validation_results = []
        for i, episode_data in enumerate(episodes_data):
            if hasattr(aggregator, 'validate_data'):
                validation = aggregator.validate_data(episode_data)
                if not validation['valid']:
                    logger.warning(f"Episode {i} validation issues: {validation['errors']}")
                validation_results.append({
                    'episode_index': i,
                    'episode_id': episode_data.get('episode_id', i),
                    **validation
                })
        
        # Calculate run-level statistics
        logger.info(f"Generating summary for {len(episodes_data)} episodes")
        run_statistics = aggregator.calculate_run_stats(episodes_data)
        
        # Collect performance metrics
        performance_metrics = aggregator.get_performance_metrics()
        
        # Calculate processing timing
        processing_end_time = time.time()
        total_processing_time = processing_end_time - processing_start_time
        
        # Compile comprehensive summary
        summary = {
            'metadata': {
                'generation_timestamp': processing_end_time,
                'processing_time_seconds': total_processing_time,
                'episodes_processed': len(episodes_data),
                'aggregator_version': getattr(aggregator, '__version__', '1.0.0'),
                'analysis_module_version': __version__
            },
            'run_statistics': run_statistics,
            'performance_metrics': performance_metrics,
            'episode_count': len(episodes_data),
            'processing_time': total_processing_time,
            'validation_results': {
                'episodes_validated': len(validation_results),
                'validation_issues': sum(1 for v in validation_results if not v['valid']),
                'overall_data_quality': 1.0 - (sum(1 for v in validation_results if not v['valid']) / len(validation_results)),
                'detailed_validation': validation_results if len(validation_results) < 100 else validation_results[:100]  # Limit for large datasets
            },
            'summary_statistics': {
                'episodes_per_second': len(episodes_data) / total_processing_time if total_processing_time > 0 else 0,
                'average_episode_processing_time': total_processing_time / len(episodes_data) if episodes_data else 0,
                'meets_performance_target': total_processing_time / len(episodes_data) <= 0.033 if episodes_data else True,  # 33ms target
                'memory_efficiency': performance_metrics.get('memory_efficiency', 1.0)
            }
        }
        
        # Export summary if output path specified
        export_status = False
        if output_path:
            try:
                export_format = export_options.get('format', 'json')
                export_success = aggregator.export_summary(
                    output_path=output_path,
                    format=export_format,
                    **export_options
                )
                
                if export_success:
                    export_status = True
                    logger.info(f"Summary exported to {output_path}")
                    summary['export_info'] = {
                        'output_path': output_path,
                        'format': export_format,
                        'export_options': export_options,
                        'export_timestamp': time.time()
                    }
                else:
                    logger.warning(f"Summary export to {output_path} failed")
                    
            except Exception as e:
                logger.error(f"Error during summary export: {e}")
                summary['export_error'] = str(e)
        
        summary['export_status'] = export_status
        
        # Performance assessment and recommendations
        performance_assessment = []
        if summary['summary_statistics']['meets_performance_target']:
            performance_assessment.append("✓ Processing meets 33ms/step performance target")
        else:
            performance_assessment.append("⚠ Processing exceeds 33ms/step performance target")
            performance_assessment.append("Consider optimizing metrics_definitions or enabling parallel processing")
        
        if summary['validation_results']['overall_data_quality'] >= 0.95:
            performance_assessment.append("✓ Excellent data quality - minimal validation issues")
        elif summary['validation_results']['overall_data_quality'] >= 0.80:
            performance_assessment.append("○ Good data quality - minor validation issues detected")
        else:
            performance_assessment.append("⚠ Data quality issues detected - review validation results")
        
        summary['performance_assessment'] = performance_assessment
        
        logger.info(f"Summary generation completed in {total_processing_time:.3f}s")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        if isinstance(e, (ValueError, AnalysisModuleError)):
            raise
        else:
            raise AnalysisModuleError(f"Summary generation failed: {e}") from e


def generate_summary_report(
    episodes_data: List[Dict[str, Any]], 
    output_path: str,
    config: Optional[Union[Dict[str, Any], StatsAggregatorConfig]] = None,
    **report_options: Any
) -> bool:
    """
    Generate and export comprehensive summary report with automatic aggregator creation.
    
    This convenience function provides a one-step solution for generating comprehensive
    summary reports from episode data. It automatically creates an appropriately configured
    statistics aggregator, processes the data, and exports results in the specified format.
    This is the recommended approach for simple analysis workflows and automated reporting.
    
    Args:
        episodes_data: List of episode data dictionaries for analysis
        output_path: File system path for summary report export
        config: Optional configuration for statistics aggregator (uses defaults if None)
        **report_options: Additional options for report generation and export
        
    Returns:
        bool: True if report generation and export completed successfully, False otherwise
        
    Examples:
        Simple automated report generation:
        >>> success = generate_summary_report(
        ...     episodes_data, 
        ...     './results/experiment_report.json'
        ... )
        >>> if success:
        ...     print("Report generated successfully")
        
        Customized report with specific metrics:
        >>> custom_config = {
        ...     'metrics_definitions': {
        ...         'trajectory': ['mean', 'std', 'efficiency', 'tortuosity'],
        ...         'concentration': ['detection_rate', 'mean', 'max']
        ...     },
        ...     'aggregation_levels': ['episode', 'run'],
        ...     'performance_tracking': True
        ... }
        >>> success = generate_summary_report(
        ...     episodes_data,
        ...     './results/detailed_report.json',
        ...     config=custom_config,
        ...     format='json',
        ...     indent=2
        ... )
    """
    try:
        # Use default configuration if none provided
        if config is None:
            config = {
                'metrics_definitions': {
                    'trajectory': ['mean', 'std', 'total_distance', 'displacement_efficiency'],
                    'concentration': ['mean', 'std', 'detection_rate'],
                    'speed': ['mean', 'max', 'total_distance']
                },
                'aggregation_levels': ['episode', 'run'],
                'performance_tracking': True,
                'output_format': 'json'
            }
        
        # Create aggregator
        aggregator = create_stats_aggregator(config)
        
        # Generate summary
        summary = generate_summary(
            aggregator,
            episodes_data,
            output_path=output_path,
            **report_options
        )
        
        return summary.get('export_status', False)
        
    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")
        return False


# Export public API components with comprehensive functionality
__all__ = [
    # Core classes and interfaces
    'StatsAggregator',
    'StatsAggregatorProtocol', 
    'StatsAggregatorConfig',
    
    # Factory and creation functions
    'create_stats_aggregator',
    'create_from_config',
    
    # Summary generation functions
    'generate_summary',
    'generate_summary_report',
    
    # Statistical calculation utilities
    'calculate_basic_stats',
    'calculate_advanced_stats',
    
    # Validation and compatibility utilities
    'validate_recorder_compatibility',
    'check_analysis_dependencies',
    
    # Module metadata and version
    '__version__',
    
    # Exception classes for error handling
    'AnalysisModuleError',
    'ConfigurationError', 
    'RecorderCompatibilityError'
]

# Module initialization and dependency validation
try:
    # Validate core dependencies on import
    dep_status = check_analysis_dependencies()
    
    if not dep_status['all_available']:
        missing_deps = [name for name, available in dep_status['core_dependencies'].items() if not available]
        warnings.warn(
            f"Missing core dependencies: {missing_deps}. "
            f"Some analysis functionality may be limited.",
            ImportWarning,
            stacklevel=2
        )
    
    # Log successful initialization
    logger.debug(f"Analysis module initialized successfully (version {__version__})")
    
    # Log optional dependency status
    if not HYDRA_AVAILABLE:
        logger.info("Hydra configuration features disabled - using fallback implementations")
    
    optional_count = sum(dep_status['optional_dependencies'].values())
    total_optional = len(dep_status['optional_dependencies'])
    logger.debug(f"Optional dependencies: {optional_count}/{total_optional} available")

except Exception as e:
    # Graceful degradation on initialization errors
    warnings.warn(
        f"Analysis module initialization encountered issues: {e}. "
        f"Some features may be limited.",
        ImportWarning,
        stacklevel=2
    )