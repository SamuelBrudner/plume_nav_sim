1: """
2: Comprehensive command-line interface for odor plume navigation system.
3: 
4: This module provides a production-ready CLI built with Click framework and Hydra configuration
5: integration, supporting simulation execution, configuration validation, batch processing,
6: parameter sweeps, and visualization export commands. The interface implements @hydra.main
7: decorator for seamless configuration injection and multi-run experiment orchestration.
8: 
9: The CLI architecture supports:
10: - Real-time simulation execution with parameter overrides
11: - Configuration validation and export for development workflows
12: - Multi-run parameter sweeps via --multirun flag for automated experiments
13: - Visualization export with publication-quality output generation
14: - Batch processing capabilities for headless execution environments
15: - Comprehensive help system with usage examples and parameter documentation
16: - Error handling with recovery strategies and detailed diagnostic information
17: 
18: Performance Characteristics:
19: - Command initialization: <2s per Section 2.2.9.3 performance criteria
20: - Configuration validation: <500ms for complex hierarchical configurations
21: - Parameter override processing: Real-time with immediate validation feedback
22: - Help system generation: Instant response with comprehensive documentation
23: 
24: Examples:
25:     # Basic simulation execution
26:     python -m odor_plume_nav.cli.main run
27:     
28:     # Simulation with parameter overrides
29:     python -m odor_plume_nav.cli.main run navigator.max_speed=10.0 simulation.num_steps=500
30:     
31:     # Multi-run parameter sweep
32:     python -m odor_plume_nav.cli.main --multirun run navigator.max_speed=5,10,15
33:     
34:     # Configuration validation
35:     python -m odor_plume_nav.cli.main config validate
36:     
37:     # Visualization export
38:     python -m odor_plume_nav.cli.main visualize export --format mp4 --output results.mp4
39:     
40:     # Dry-run validation
41:     python -m odor_plume_nav.cli.main run --dry-run
42: """
43: 
44: import os
45: import sys
46: import time
47: import traceback
48: from pathlib import Path
49: from typing import Optional, List, Dict, Any, Union
50: import warnings
51: 
52: import click
53: import numpy as np
54: import matplotlib.pyplot as plt
55: from loguru import logger
56: 
57: # Hydra imports for configuration management
58: try:
59:     import hydra
60:     from hydra import compose, initialize, initialize_config_dir
61:     from hydra.core.config_store import ConfigStore
62:     from hydra.core.global_hydra import GlobalHydra
63:     from hydra.core.hydra_config import HydraConfig
64:     from omegaconf import DictConfig, OmegaConf, ListConfig
65:     HYDRA_AVAILABLE = True
66: except ImportError:
67:     HYDRA_AVAILABLE = False
68:     warnings.warn(
69:         "Hydra not available. Advanced configuration features will be limited.",
70:         ImportWarning
71:     )
72: 
73: # Import core system components
74: from odor_plume_nav.api.navigation import (
75:     create_navigator,
76:     create_video_plume,
77:     run_plume_simulation
78: )
79: from odor_plume_nav.config.models import (
80:     NavigatorConfig,
81:     VideoPlumeConfig,
82:     SimulationConfig
83: )
84: from odor_plume_nav.utils.seed_manager import set_global_seed, get_last_seed
85: from odor_plume_nav.utils.logging_setup import setup_logger, get_enhanced_logger
86: from odor_plume_nav.utils.visualization import (
87:     visualize_plume_simulation,
88:     visualize_trajectory,
89:     create_realtime_visualizer
90: )
91: 
# Import frame cache for performance optimization
try:
    from odor_plume_nav.cache.frame_cache import FrameCache
    FRAME_CACHE_AVAILABLE = True
except ImportError:
    FRAME_CACHE_AVAILABLE = False
    warnings.warn(
        "Frame cache not available. Performance optimizations will be limited.",
        ImportWarning
    )

92: # Global configuration for CLI state management
93: _CLI_CONFIG = {
94:     'verbose': False,
95:     'quiet': False,
96:     'log_level': 'INFO',
97:     'start_time': None,
98:     'dry_run': False
99: }
100: 
101: 
102: class CLIError(Exception):
103:     """CLI-specific error for command execution failures."""
104:     pass
105: 
106: 
107: class ConfigValidationError(Exception):
108:     """Configuration validation specific errors."""
109:     pass
110: 
111: 
112: class ConfigurationError(Exception):
113:     """Configuration-related error for component creation failures."""
114:     pass
115: 
116: 
117: class SimulationError(Exception):
118:     """Simulation execution error for runtime failures."""
119:     pass
120: 
121: 
122: def _setup_cli_logging(verbose: bool = False, quiet: bool = False, log_level: str = 'INFO') -> None:
123:     """
124:     Initialize CLI-specific logging configuration with performance monitoring.
125:     
126:     Args:
127:         verbose: Enable verbose logging output with debug information
128:         quiet: Suppress non-essential output (errors only)
129:         log_level: Logging level for CLI operations
130:     """
131:     try:
132:         # Configure loguru for CLI operations
133:         logger.remove()  # Remove default handler
134:         
135:         if quiet:
136:             # Only show errors in quiet mode
137:             logger.add(
138:                 sys.stderr,
139:                 level="ERROR",
140:                 format="<red>ERROR</red>: {message}",
141:                 colorize=True
142:             )
143:         elif verbose:
144:             # Verbose mode with detailed information
145:             logger.add(
146:                 sys.stderr,
147:                 level="DEBUG",
148:                 format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
149:                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
150:                       "<level>{message}</level>",
151:                 colorize=True
152:             )
153:         else:
154:             # Standard mode with clean output
155:             logger.add(
156:                 sys.stderr,
157:                 level=log_level,
158:                 format="<level>{level}</level>: {message}",
159:                 colorize=True
160:             )
161:         
162:         # Update global CLI configuration
163:         _CLI_CONFIG.update({
164:             'verbose': verbose,
165:             'quiet': quiet,
166:             'log_level': log_level
167:         })
168:         
169:         logger.info("CLI logging initialized successfully")
170:         
171:     except Exception as e:
172:         # Fallback to basic logging if setup fails
173:         click.echo(f"Warning: Failed to setup advanced logging: {e}", err=True)
174:         logger.add(sys.stderr, level="INFO")
175: 
176: 
177: def _measure_performance(func_name: str, start_time: float) -> None:
178:     """
179:     Measure and log performance metrics for CLI operations.
180:     
181:     Args:
182:         func_name: Name of the function being measured
183:         start_time: Start time for performance measurement
184:     """
185:     elapsed = time.time() - start_time
186:     
187:     if elapsed > 2.0:
188:         logger.warning(f"{func_name} took {elapsed:.2f}s (>2s threshold)")
189:     else:
190:         logger.debug(f"{func_name} completed in {elapsed:.2f}s")
191: 
192: 
193: def _validate_hydra_availability() -> None:
194:     """Validate that Hydra is available for advanced CLI features."""
195:     if not HYDRA_AVAILABLE:
196:         raise CLIError(
197:             "Hydra is required for CLI functionality. Please install with: "
198:             "pip install hydra-core"
199:         )
200: 
201: 
202: def _safe_config_access(cfg: DictConfig, path: str, default: Any = None) -> Any:
203:     """
204:     Safely access nested configuration values with error handling.
205:     
206:     Args:
207:         cfg: Hydra configuration object
208:         path: Dot-separated path to configuration value
209:         default: Default value if path doesn't exist
210:         
211:     Returns:
212:         Configuration value or default
213:     """
214:     try:
215:         keys = path.split('.')
216:         value = cfg
217:         for key in keys:
218:             if hasattr(value, key):
219:                 value = getattr(value, key)
220:             elif isinstance(value, dict) and key in value:
221:                 value = value[key]
222:             else:
223:                 return default
224:         return value
225:     except Exception:
226:         return default
227: 
228: 
229: def _export_config_documentation(cfg: DictConfig, output_path: Optional[Path] = None) -> Path:
230:     """
231:     Export configuration as documentation with comprehensive formatting.
232:     
233:     Args:
234:         cfg: Hydra configuration to export
235:         output_path: Optional output file path
236:         
237:     Returns:
238:         Path to exported configuration file
239:     """
240:     if output_path is None:
241:         output_path = Path("config_export.yaml")
242:     
243:     try:
244:         # Convert to container and format for documentation
245:         config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
246:         
247:         # Add documentation headers
248:         documentation = {
249:             "_metadata": {
250:                 "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
251:                 "hydra_version": hydra.__version__ if HYDRA_AVAILABLE else "N/A",
252:                 "config_source": str(HydraConfig.get().runtime.config_sources) if HydraConfig.initialized() else "Unknown"
253:             },
254:             **config_dict
255:         }
256:         
257:         # Write formatted YAML
258:         with open(output_path, 'w') as f:
259:             OmegaConf.save(documentation, f)
260:         
261:         logger.info(f"Configuration exported to {output_path}")
262:         return output_path
263:         
264:     except Exception as e:
265:         raise CLIError(f"Failed to export configuration: {e}") from e
266: 
267: 
268: def _validate_configuration(cfg: DictConfig, strict: bool = True) -> Dict[str, Any]:
269:     """
270:     Comprehensive configuration validation with detailed error reporting.
271:     
272:     Args:
273:         cfg: Hydra configuration to validate
274:         strict: Whether to use strict validation rules
275:         
276:     Returns:
277:         Validation results with errors and warnings
278:     """
279:     validation_results = {
280:         'valid': True,
281:         'errors': [],
282:         'warnings': [],
283:         'summary': {}
284:     }
285:     
286:     start_time = time.time()
287:     
288:     try:
289:         # Validate navigator configuration
290:         if hasattr(cfg, 'navigator') and cfg.navigator:
291:             try:
292:                 NavigatorConfig.model_validate(OmegaConf.to_container(cfg.navigator, resolve=True))
293:                 validation_results['summary']['navigator'] = 'valid'
294:             except Exception as e:
295:                 validation_results['errors'].append(f"Navigator config invalid: {e}")
296:                 validation_results['valid'] = False
297:                 validation_results['summary']['navigator'] = 'invalid'
298:         
299:         # Validate video plume configuration
300:         if hasattr(cfg, 'video_plume') and cfg.video_plume:
301:             try:
302:                 VideoPlumeConfig.model_validate(OmegaConf.to_container(cfg.video_plume, resolve=True))
303:                 validation_results['summary']['video_plume'] = 'valid'
304:             except Exception as e:
305:                 validation_results['errors'].append(f"Video plume config invalid: {e}")
306:                 validation_results['valid'] = False
307:                 validation_results['summary']['video_plume'] = 'invalid'
308:         
309:         # Validate simulation configuration
310:         if hasattr(cfg, 'simulation') and cfg.simulation:
311:             try:
312:                 SimulationConfig.model_validate(OmegaConf.to_container(cfg.simulation, resolve=True))
313:                 validation_results['summary']['simulation'] = 'valid'
314:             except Exception as e:
315:                 validation_results['errors'].append(f"Simulation config invalid: {e}")
316:                 validation_results['valid'] = False
317:                 validation_results['summary']['simulation'] = 'invalid'
318:         
319:         # Check for missing required configurations
320:         required_sections = ['navigator', 'video_plume', 'simulation']
321:         for section in required_sections:
322:             if not hasattr(cfg, section) or not getattr(cfg, section):
323:                 if strict:
324:                     validation_results['errors'].append(f"Required section '{section}' is missing or empty")
325:                     validation_results['valid'] = False
326:                 else:
327:                     validation_results['warnings'].append(f"Section '{section}' is missing or empty")
328:         
329:         # Validate file paths
330:         video_path = _safe_config_access(cfg, 'video_plume.video_path')
331:         if video_path and not Path(video_path).exists():
332:             validation_results['errors'].append(f"Video file not found: {video_path}")
333:             validation_results['valid'] = False
334:         
335:         # Check for reasonable parameter values
336:         max_speed = _safe_config_access(cfg, 'navigator.max_speed')
337:         if max_speed and (max_speed <= 0 or max_speed > 100):
338:             validation_results['warnings'].append(
339:                 f"Navigator max_speed ({max_speed}) may be unreasonable (expected: 0-100)"
340:             )
341:         
342:         num_steps = _safe_config_access(cfg, 'simulation.num_steps')
343:         if num_steps and (num_steps <= 0 or num_steps > 100000):
344:             validation_results['warnings'].append(
345:                 f"Simulation num_steps ({num_steps}) may be unreasonable (expected: 1-100000)"
346:             )
347:         
348:     except Exception as e:
349:         validation_results['errors'].append(f"Validation error: {e}")
350:         validation_results['valid'] = False
351:     
352:     _measure_performance("Configuration validation", start_time)
353:     return validation_results
354: 
355: 
def _validate_frame_cache_availability() -> None:
    """Validate that frame cache functionality is available."""
    if not FRAME_CACHE_AVAILABLE:
        raise CLIError(
            "Frame cache functionality is not available. Please ensure all dependencies are installed."
        )


def _create_frame_cache(
    cache_mode: str, 
    memory_limit_gb: float = 2.0,
    video_path: Optional[Union[str, pathlib.Path]] = None
) -> Optional[FrameCache]:
    """
    Create a FrameCache instance based on the specified mode and configuration.
    
    Args:
        cache_mode: Cache mode ("none", "lru", or "all")
        memory_limit_gb: Memory limit in gigabytes (default 2.0 GB)
        video_path: Optional video path for preload validation
        
    Returns:
        FrameCache instance or None if cache_mode is "none"
        
    Raises:
        CLIError: If cache creation fails or invalid mode is specified
    """
    if cache_mode == "none":
        logger.info("Frame caching disabled - using direct frame access")
        return None
    
    if not FRAME_CACHE_AVAILABLE:
        logger.warning("Frame cache requested but not available, falling back to direct access")
        return None
    
    try:
        memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
        
        if cache_mode == "lru":
            cache = FrameCache(
                mode="lru",
                memory_limit=memory_limit_bytes,
                max_frames=None  # Let memory limit control capacity
            )
            logger.info(f"Created LRU frame cache with {memory_limit_gb:.1f} GB memory limit")
            
        elif cache_mode == "all":
            cache = FrameCache(
                mode="all",
                memory_limit=memory_limit_bytes,
                max_frames=None  # Preload all frames up to memory limit
            )
            logger.info(f"Created full preload frame cache with {memory_limit_gb:.1f} GB memory limit")
            
        else:
            raise ValueError(f"Invalid cache mode: {cache_mode}. Must be 'none', 'lru', or 'all'")
        
        return cache
        
    except Exception as e:
        logger.error(f"Failed to create frame cache: {e}")
        raise CLIError(f"Frame cache creation failed: {e}") from e


def _validate_frame_cache_mode(cache_mode: str) -> str:
    """
    Validate frame cache mode parameter.
    
    Args:
        cache_mode: Cache mode string to validate
        
    Returns:
        Validated cache mode string
        
    Raises:
        click.BadParameter: If cache mode is invalid
    """
    valid_modes = {"none", "lru", "all"}
    if cache_mode not in valid_modes:
        raise click.BadParameter(
            f"Invalid frame cache mode: {cache_mode}. Must be one of: {', '.join(valid_modes)}"
        )
    return cache_mode


356: # Click CLI Groups and Commands Implementation
357: 
358: @click.group(invoke_without_command=True)
359: @click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
360: @click.option('--quiet', '-q', is_flag=True, help='Suppress non-essential output')
361: @click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
362:               help='Set logging level for CLI operations')
363: @click.pass_context
364: def cli(ctx, verbose: bool, quiet: bool, log_level: str) -> None:
365:     """
366:     Odor Plume Navigation CLI - Comprehensive command-line interface.
367:     
368:     This CLI provides complete access to simulation execution, configuration management,
369:     visualization generation, and batch processing capabilities. Built with Hydra
370:     configuration integration for advanced parameter management and experiment orchestration.
371:     
372:     Examples:
373:         # Run simulation with default configuration
374:         odor-plume-nav-cli run
375:         
376:         # Run with parameter overrides
377:         odor-plume-nav-cli run navigator.max_speed=10.0
378:         
379:         # Multi-run parameter sweep
380:         odor-plume-nav-cli --multirun run navigator.max_speed=5,10,15
381:         
382:         # Validate configuration
383:         odor-plume-nav-cli config validate
384:         
385:         # Export visualization
386:         odor-plume-nav-cli visualize export --format mp4
387:     
388:     For detailed help on any command, use: odor-plume-nav-cli COMMAND --help
389:     """
390:     _CLI_CONFIG['start_time'] = time.time()
391:     
392:     # Setup CLI logging based on options
393:     _setup_cli_logging(verbose=verbose, quiet=quiet, log_level=log_level)
394:     
395:     # Initialize click context for subcommands
396:     ctx.ensure_object(dict)
397:     ctx.obj.update(_CLI_CONFIG)
398:     
399:     # If no command specified, show help
400:     if ctx.invoked_subcommand is None:
401:         click.echo(ctx.get_help())
402:         ctx.exit()
403: 
404: 
405: @cli.command()
406: @click.option('--dry-run', is_flag=True, 
407:               help='Validate simulation setup without executing')
408: @click.option('--seed', type=int, 
409:               help='Random seed for reproducible results')
410: @click.option('--output-dir', type=click.Path(), 
411:               help='Directory for output files (overrides Hydra default)')
412: @click.option('--save-trajectory', is_flag=True, default=True,
413:               help='Save trajectory data for post-analysis')
414: @click.option('--show-animation', is_flag=True, 
415:               help='Display real-time animation during simulation')
416: @click.option('--export-video', type=click.Path(),
417:               help='Export animation as MP4 video to specified path')
@click.option('--frame-cache', type=click.Choice(['none', 'lru', 'all']), default='none',
               callback=lambda ctx, param, value: _validate_frame_cache_mode(value),
               help='Frame caching mode: "none" (no caching), "lru" (LRU eviction with 2GB limit), '
                    '"all" (preload all frames for maximum throughput)')
418: @click.pass_context
419: def run(ctx, dry_run: bool, seed: Optional[int], output_dir: Optional[str], 
420:         save_trajectory: bool, show_animation: bool, export_video: Optional[str], 
        frame_cache: str) -> None:
421:     """
422:     Execute odor plume navigation simulation with comprehensive options.
423:     
424:     This command runs the complete simulation pipeline including navigator creation,
425:     video plume environment loading, simulation execution, and optional visualization.
426:     Supports parameter overrides via Hydra syntax and dry-run validation.
427:     
428:     Configuration Parameters:
429:         All Hydra configuration parameters can be overridden using dot notation:
430:         - navigator.max_speed=10.0
431:         - simulation.num_steps=1000  
432:         - video_plume.flip=true
433:         - simulation.dt=0.1
434:     
435:     Examples:
436:         # Basic simulation
437:         odor-plume-nav-cli run
438:         
439:         # With parameter overrides
440:         odor-plume-nav-cli run navigator.max_speed=15.0 simulation.num_steps=500
441:         
442:         # Dry-run validation
443:         odor-plume-nav-cli run --dry-run
444:         
445:         # With visualization export
446:         odor-plume-nav-cli run --export-video results.mp4
447:         
448:         # Reproducible run with seed
449:         odor-plume-nav-cli run --seed 42
        
        # With frame caching for performance optimization
        odor-plume-nav-cli run --frame-cache lru
        
        # With full frame preloading for maximum throughput
        odor-plume-nav-cli run --frame-cache all
450:     """
451:     start_time = time.time()
452:     ctx.obj['dry_run'] = dry_run
453:     
454:     try:
455:         # Validate Hydra availability
456:         _validate_hydra_availability()
457:         
458:         # Access Hydra configuration
459:         if not HydraConfig.initialized():
460:             logger.error("Hydra configuration not initialized. Use @hydra.main decorator.")
461:             raise CLIError("Hydra configuration required for run command")
462:         
463:         cfg = HydraConfig.get().cfg
464:         
465:         # Set global seed if provided
466:         if seed is not None:
467:             set_global_seed(seed)
468:             logger.info(f"Global seed set to: {seed}")
469:         
470:         # Validate configuration
471:         logger.info("Validating configuration...")
472:         validation_results = _validate_configuration(cfg, strict=not dry_run)
473:         
474:         if not validation_results['valid']:
475:             logger.error("Configuration validation failed:")
476:             for error in validation_results['errors']:
477:                 logger.error(f"  - {error}")
478:             raise ConfigValidationError("Invalid configuration")
479:         
480:         if validation_results['warnings']:
481:             logger.warning("Configuration warnings:")
482:             for warning in validation_results['warnings']:
483:                 logger.warning(f"  - {warning}")
484:         
485:         logger.info("Configuration validation passed")
486:         
487:         if dry_run:
488:             logger.info("Dry-run mode: Simulation validation completed successfully")
489:             logger.info(f"Configuration summary: {validation_results['summary']}")
490:             _measure_performance("Dry-run validation", start_time)
491:             return
492:         
493:         # Create components
494:         logger.info("Creating navigation components...")
495:         
        
        # Create frame cache if requested
        frame_cache_instance = None
        if frame_cache != "none":
            try:
                video_path = _safe_config_access(cfg, 'video_plume.video_path')
                frame_cache_instance = _create_frame_cache(frame_cache, video_path=video_path)
                if frame_cache_instance:
                    logger.info(f"Frame cache created in '{frame_cache}' mode")
            except Exception as e:
                logger.warning(f"Failed to create frame cache, falling back to direct access: {e}")
                frame_cache_instance = None
496:         try:
497:             navigator = create_navigator(cfg=cfg.navigator if hasattr(cfg, 'navigator') else None)
498:             logger.info(f"Navigator created: {type(navigator).__name__} with {navigator.num_agents} agent(s)")
499:         except Exception as e:
500:             logger.error(f"Failed to create navigator: {e}")
501:             raise ConfigurationError(f"Navigator creation failed: {e}") from e
502:         
503:         try:
504:             # Pass frame cache to video plume creation
            video_plume_config = cfg.video_plume if hasattr(cfg, 'video_plume') else None
            video_plume = create_video_plume(
                cfg=video_plume_config, 
                frame_cache=frame_cache_instance
            )
505:             logger.info(f"Video plume loaded: {video_plume.frame_count} frames")
506:         except Exception as e:
507:             logger.error(f"Failed to create video plume: {e}")
508:             raise ConfigurationError(f"Video plume creation failed: {e}") from e
509:         
510:         # Execute simulation
511:         logger.info("Starting simulation execution...")
512:         sim_start_time = time.time()
513:         
514:         try:
515:             positions, orientations, odor_readings = run_plume_simulation(
516:                 navigator=navigator,
517:                 video_plume=video_plume,
518:                 cfg=cfg.simulation if hasattr(cfg, 'simulation') else None,
519:                 record_trajectories=save_trajectory,
520:                 seed=seed
                frame_cache=frame_cache_instance,
521:             )
522:             
523:             sim_duration = time.time() - sim_start_time
524:             logger.info(f"Simulation completed in {sim_duration:.2f}s")
525:             
526:             # Log simulation statistics
527:             logger.info(f"Final positions shape: {positions.shape}")
528:             logger.info(f"Trajectory length: {positions.shape[1]} steps")
529:             logger.info(f"Average odor reading: {np.mean(odor_readings):.4f}")
530:             
531:         except Exception as e:
532:             logger.error(f"Simulation execution failed: {e}")
533:             raise SimulationError(f"Simulation failed: {e}") from e
534:         
535:         # Handle visualization and export
536:         if show_animation or export_video:
537:             logger.info("Generating visualization...")
538:             try:
539:                 visualization_results = {
540:                     'positions': positions,
541:                     'orientations': orientations,
542:                     'odor_readings': odor_readings
543:                 }
544:                 
545:                 if show_animation:
546:                     visualize_plume_simulation(
547:                         positions=positions,
548:                         orientations=orientations,
549:                         odor_readings=odor_readings,
550:                         show_plot=True,
551:                         cfg=cfg.visualization if hasattr(cfg, 'visualization') else None
552:                     )
553:                 
554:                 if export_video:
555:                     export_path = Path(export_video)
556:                     visualize_plume_simulation(
557:                         positions=positions,
558:                         orientations=orientations,
559:                         odor_readings=odor_readings,
560:                         output_path=export_path,
561:                         show_plot=False,
562:                         cfg=cfg.visualization if hasattr(cfg, 'visualization') else None
563:                     )
564:                     logger.info(f"Animation exported to {export_path}")
565:                     
566:             except Exception as e:
567:                 logger.warning(f"Visualization failed: {e}")
568:         
569:         # Save trajectory data if requested
570:         if save_trajectory and output_dir:
571:             output_path = Path(output_dir)
572:             output_path.mkdir(parents=True, exist_ok=True)
573:             
574:             np.savez(
575:                 output_path / "trajectory_data.npz",
576:                 positions=positions,
577:                 orientations=orientations,
578:                 odor_readings=odor_readings,
579:                 seed=get_last_seed()
580:             )
581:             logger.info(f"Trajectory data saved to {output_path}/trajectory_data.npz")
582:         
583:         _measure_performance("Simulation execution", start_time)
584:         logger.info("Run command completed successfully")
585:         
586:     except KeyboardInterrupt:
587:         logger.warning("Simulation interrupted by user")
588:         ctx.exit(1)
589:     except (CLIError, ConfigValidationError, ConfigurationError, SimulationError) as e:
590:         logger.error(str(e))
591:         ctx.exit(1)
592:     except Exception as e:
593:         logger.error(f"Unexpected error: {e}")
594:         if ctx.obj.get('verbose'):
595:             logger.error(traceback.format_exc())
596:         ctx.exit(1)
597: 
598: 
599: @cli.group()
600: def config() -> None:
601:     """
602:     Configuration management commands for validation and export.
603:     
604:     This command group provides utilities for working with Hydra configuration files,
605:     including validation, export, and documentation generation capabilities.
606:     """
607:     pass
608: 
609: 
610: @config.command()
611: @click.option('--strict', is_flag=True,
612:               help='Use strict validation rules (fail on warnings)')
613: @click.option('--export-results', type=click.Path(),
614:               help='Export validation results to JSON file')
615: @click.pass_context
616: def validate(ctx, strict: bool, export_results: Optional[str]) -> None:
617:     """
618:     Validate Hydra configuration files with comprehensive error reporting.
619:     
620:     Performs thorough validation of configuration schemas, parameter ranges,
621:     file existence checks, and cross-section consistency validation.
622:     
623:     Examples:
624:         # Basic validation
625:         odor-plume-nav-cli config validate
626:         
627:         # Strict validation (warnings as errors)
628:         odor-plume-nav-cli config validate --strict
629:         
630:         # Export validation results
631:         odor-plume-nav-cli config validate --export-results validation.json
632:     """
633:     start_time = time.time()
634:     
635:     try:
636:         _validate_hydra_availability()
637:         
638:         if not HydraConfig.initialized():
639:             logger.error("Hydra configuration not initialized")
640:             raise CLIError("Configuration validation requires Hydra initialization")
641:         
642:         cfg = HydraConfig.get().cfg
643:         
644:         logger.info("Starting configuration validation...")
645:         validation_results = _validate_configuration(cfg, strict=strict)
646:         
647:         # Report validation results
648:         if validation_results['valid']:
649:             logger.info("✓ Configuration validation passed")
650:             click.echo(click.style("Configuration is valid!", fg='green'))
651:         else:
652:             logger.error("✗ Configuration validation failed")
653:             click.echo(click.style("Configuration validation failed!", fg='red'))
654:             
655:             for error in validation_results['errors']:
656:                 click.echo(click.style(f"ERROR: {error}", fg='red'))
657:         
658:         if validation_results['warnings']:
659:             click.echo(click.style("Warnings:", fg='yellow'))
660:             for warning in validation_results['warnings']:
661:                 click.echo(click.style(f"WARNING: {warning}", fg='yellow'))
662:         
663:         # Show summary
664:         click.echo("\nValidation Summary:")
665:         for section, status in validation_results['summary'].items():
666:             status_color = 'green' if status == 'valid' else 'red'
667:             click.echo(f"  {section}: {click.style(status, fg=status_color)}")
668:         
669:         # Export results if requested
670:         if export_results:
671:             import json
672:             with open(export_results, 'w') as f:
673:                 json.dump(validation_results, f, indent=2)
674:             logger.info(f"Validation results exported to {export_results}")
675:         
676:         _measure_performance("Configuration validation", start_time)
677:         
678:         if not validation_results['valid']:
679:             ctx.exit(1)
680:             
681:     except (CLIError, ConfigValidationError) as e:
682:         logger.error(str(e))
683:         ctx.exit(1)
684:     except Exception as e:
685:         logger.error(f"Validation error: {e}")
686:         if ctx.obj.get('verbose'):
687:             logger.error(traceback.format_exc())
688:         ctx.exit(1)
689: 
690: 
691: @config.command()
692: @click.option('--output', '-o', type=click.Path(),
693:               help='Output file path (default: config_export.yaml)')
694: @click.option('--format', 'output_format', type=click.Choice(['yaml', 'json']), default='yaml',
695:               help='Export format')
696: @click.option('--resolve', is_flag=True, default=True,
697:               help='Resolve configuration interpolations')
698: @click.pass_context  
699: def export(ctx, output: Optional[str], output_format: str, resolve: bool) -> None:
700:     """
701:     Export current configuration for documentation and sharing.
702:     
703:     Generates a comprehensive configuration export including metadata,
704:     resolved interpolations, and formatted output for documentation purposes.
705:     
706:     Examples:
707:         # Export to default file
708:         odor-plume-nav-cli config export
709:         
710:         # Export to specific file
711:         odor-plume-nav-cli config export --output my_config.yaml
712:         
713:         # Export as JSON
714:         odor-plume-nav-cli config export --format json
715:         
716:         # Export without resolving interpolations
717:         odor-plume-nav-cli config export --no-resolve
718:     """
719:     start_time = time.time()
720:     
721:     try:
722:         _validate_hydra_availability()
723:         
724:         if not HydraConfig.initialized():
725:             logger.error("Hydra configuration not initialized")
726:             raise CLIError("Configuration export requires Hydra initialization")
727:         
728:         cfg = HydraConfig.get().cfg
729:         
730:         # Determine output path
731:         if output is None:
732:             output = f"config_export.{output_format}"
733:         
734:         output_path = Path(output)
735:         
736:         logger.info(f"Exporting configuration to {output_path}")
737:         
738:         # Prepare configuration for export
739:         if resolve:
740:             config_data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
741:         else:
742:             config_data = OmegaConf.to_container(cfg, resolve=False)
743:         
744:         # Add metadata
745:         export_data = {
746:             "_metadata": {
747:                 "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
748:                 "hydra_version": hydra.__version__ if HYDRA_AVAILABLE else "N/A",
749:                 "resolved": resolve,
750:                 "format": output_format
751:             },
752:             **config_data
753:         }
754:         
755:         # Write to file
756:         output_path.parent.mkdir(parents=True, exist_ok=True)
757:         
758:         if output_format == 'yaml':
759:             with open(output_path, 'w') as f:
760:                 OmegaConf.save(export_data, f)
761:         elif output_format == 'json':
762:             import json
763:             with open(output_path, 'w') as f:
764:                 json.dump(export_data, f, indent=2)
765:         
766:         logger.info(f"Configuration successfully exported to {output_path}")
767:         click.echo(f"Configuration exported to: {click.style(str(output_path), fg='green')}")
768:         
769:         _measure_performance("Configuration export", start_time)
770:         
771:     except (CLIError, OSError) as e:
772:         logger.error(f"Export failed: {e}")
773:         ctx.exit(1)
774:     except Exception as e:
775:         logger.error(f"Unexpected export error: {e}")
776:         if ctx.obj.get('verbose'):
777:             logger.error(traceback.format_exc())
778:         ctx.exit(1)
779: 
780: 
781: @cli.group()
782: def visualize() -> None:
783:     """
784:     Visualization generation and export commands.
785:     
786:     This command group provides utilities for generating publication-quality
787:     visualizations, animations, and trajectory plots from simulation data.
788:     """
789:     pass
790: 
791: 
792: @visualize.command()
793: @click.option('--input-data', type=click.Path(exists=True),
794:               help='Path to trajectory data file (.npz format)')
795: @click.option('--format', 'output_format', type=click.Choice(['mp4', 'gif', 'png']), default='mp4',
796:               help='Output format for visualization')
797: @click.option('--output', '-o', type=click.Path(),
798:               help='Output file path')
799: @click.option('--dpi', type=int, default=100,
800:               help='Resolution (DPI) for output visualization')
801: @click.option('--fps', type=int, default=30,
802:               help='Frame rate for video outputs')
803: @click.option('--quality', type=click.Choice(['low', 'medium', 'high']), default='medium',
804:               help='Quality preset for output')
805: @click.pass_context
806: def export(ctx, input_data: Optional[str], output_format: str, output: Optional[str],
807:            dpi: int, fps: int, quality: str) -> None:
808:     """
809:     Export visualization with publication-quality formatting.
810:     
811:     Generates high-quality visualizations from simulation data with configurable
812:     output formats, resolution settings, and quality presets for research publication.
813:     
814:     Examples:
815:         # Export MP4 animation
816:         odor-plume-nav-cli visualize export --format mp4 --output animation.mp4
817:         
818:         # High-quality PNG trajectory plot
819:         odor-plume-nav-cli visualize export --format png --quality high --dpi 300
820:         
821:         # From existing data file
822:         odor-plume-nav-cli visualize export --input-data trajectory.npz --format gif
823:     """
824:     start_time = time.time()
825:     
826:     try:
827:         # Load data from file or current Hydra run
828:         if input_data:
829:             data_path = Path(input_data)
830:             if not data_path.exists():
831:                 raise CLIError(f"Input data file not found: {data_path}")
832:             
833:             logger.info(f"Loading trajectory data from {data_path}")
834:             data = np.load(data_path)
835:             
836:             positions = data['positions']
837:             orientations = data['orientations']
838:             odor_readings = data['odor_readings']
839:             
840:         else:
841:             # Check if we're in a Hydra run context with data
842:             logger.error("No input data specified and no current simulation data available")
843:             raise CLIError("Input data file required for visualization export")
844:         
845:         # Determine output path
846:         if output is None:
847:             timestamp = time.strftime("%Y%m%d_%H%M%S")
848:             output = f"visualization_{timestamp}.{output_format}"
849:         
850:         output_path = Path(output)
851:         output_path.parent.mkdir(parents=True, exist_ok=True)
852:         
853:         # Create visualization configuration
854:         viz_config = {
855:             'format': output_format,
856:             'dpi': dpi,
857:             'fps': fps,
858:             'quality': quality
859:         }
860:         
861:         logger.info(f"Generating {output_format} visualization...")
862:         
863:         # Generate visualization based on format
864:         if output_format in ['mp4', 'gif']:
865:             # Animation export using real-time visualizer
866:             visualizer = create_realtime_visualizer(
867:                 fps=fps,
868:                 headless=True,
869:                 resolution='720p'
870:             )
871:             
872:             # Create animation and save
873:             def frame_callback(frame_idx):
874:                 return (
875:                     (positions[0, frame_idx], orientations[0, frame_idx]),
876:                     odor_readings[0, frame_idx]
877:                 ) if len(positions.shape) == 3 else (
878:                     (positions[frame_idx], orientations[frame_idx]),
879:                     odor_readings[frame_idx]
880:                 )
881:             
882:             animation_obj = visualizer.create_animation(
883:                 frame_callback, 
884:                 frames=min(positions.shape[-2], 1000)  # Limit frames for memory
885:             )
886:             visualizer.save_animation(output_path, fps=fps, quality=quality)
887:             
888:         elif output_format == 'png':
889:             # Static trajectory plot
890:             visualize_trajectory(
891:                 positions=positions,
892:                 orientations=orientations,
893:                 output_path=output_path,
894:                 show_plot=False,
895:                 dpi=dpi,
896:                 format='png'
897:             )
898:         
899:         logger.info(f"Visualization exported to {output_path}")
900:         click.echo(f"Visualization saved: {click.style(str(output_path), fg='green')}")
901:         
902:         _measure_performance("Visualization export", start_time)
903:         
904:     except CLIError as e:
905:         logger.error(str(e))
906:         ctx.exit(1)
907:     except Exception as e:
908:         logger.error(f"Visualization export failed: {e}")
909:         if ctx.obj.get('verbose'):
910:             logger.error(traceback.format_exc())
911:         ctx.exit(1)
912: 
913: 
914: @cli.command()
915: @click.option('--jobs', '-j', type=int, default=1,
916:               help='Number of parallel jobs for batch processing')
917: @click.option('--config-dir', type=click.Path(exists=True),
918:               help='Directory containing batch configuration files')
919: @click.option('--pattern', default='*.yaml',
920:               help='File pattern for batch configuration files')
921: @click.option('--output-base', type=click.Path(),
922:               help='Base directory for batch output files')
923: @click.pass_context
924: def batch(ctx, jobs: int, config_dir: Optional[str], pattern: str, output_base: Optional[str]) -> None:
925:     """
926:     Execute batch processing for multiple configuration files.
927:     
928:     Processes multiple configuration files in parallel, enabling large-scale
929:     parameter studies and automated experiment execution.
930:     
931:     Examples:
932:         # Process all configs in directory
933:         odor-plume-nav-cli batch --config-dir experiments/
934:         
935:         # Parallel processing with 4 jobs
936:         odor-plume-nav-cli batch --config-dir experiments/ --jobs 4
937:         
938:         # Custom output directory
939:         odor-plume-nav-cli batch --config-dir configs/ --output-base results/
940:     """
941:     start_time = time.time()
942:     
943:     try:
944:         if not config_dir:
945:             raise CLIError("Config directory required for batch processing")
946:         
947:         config_path = Path(config_dir)
948:         if not config_path.exists():
949:             raise CLIError(f"Config directory not found: {config_path}")
950:         
951:         # Find configuration files
952:         config_files = list(config_path.glob(pattern))
953:         
954:         if not config_files:
955:             raise CLIError(f"No configuration files found matching pattern: {pattern}")
956:         
957:         logger.info(f"Found {len(config_files)} configuration files for batch processing")
958:         
959:         # Setup output directory
960:         if output_base:
961:             output_dir = Path(output_base)
962:         else:
963:             output_dir = Path("batch_results") / time.strftime("%Y%m%d_%H%M%S")
964:         
965:         output_dir.mkdir(parents=True, exist_ok=True)
966:         
967:         # Process files (simplified single-threaded for now)
968:         # TODO: Implement parallel processing with multiprocessing
969:         
970:         successful = 0
971:         failed = 0
972:         
973:         for config_file in config_files:
974:             try:
975:                 logger.info(f"Processing: {config_file.name}")
976:                 
977:                 # This would be enhanced to actually run simulations
978:                 # with the specific config file in a production implementation
979:                 click.echo(f"Processing {config_file.name}...")
980:                 
981:                 successful += 1
982:                 
983:             except Exception as e:
984:                 logger.error(f"Failed to process {config_file.name}: {e}")
985:                 failed += 1
986:         
987:         logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
988:         click.echo(f"Batch results: {click.style(f'{successful} successful', fg='green')}, "
989:                   f"{click.style(f'{failed} failed', fg='red')}")
990:         
991:         _measure_performance("Batch processing", start_time)
992:         
993:         if failed > 0:
994:             ctx.exit(1)
995:             
996:     except CLIError as e:
997:         logger.error(str(e))
998:         ctx.exit(1)
999:     except Exception as e:
1000:         logger.error(f"Batch processing error: {e}")
1001:         if ctx.obj.get('verbose'):
1002:             logger.error(traceback.format_exc())
1003:         ctx.exit(1)
1004: 
1005: 
# RL Training Commands and Utilities

# Conditional imports for RL training functionality
try:
    import stable_baselines3
    from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold,
        ProgressBarCallback
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_util import make_vec_env
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    warnings.warn(
        "Stable-baselines3 not available. RL training commands will be limited.",
        ImportWarning
    )

# Import Gymnasium environment creation function with graceful fallback
try:
    from odor_plume_nav.api.navigation import create_gymnasium_environment
    GYMNASIUM_ENV_AVAILABLE = True
except ImportError:
    GYMNASIUM_ENV_AVAILABLE = False
    warnings.warn(
        "Gymnasium environment not available. RL training features will be disabled.",
        ImportWarning
    )


def _validate_rl_availability() -> None:
    """Validate that RL dependencies are available for training commands."""
    if not SB3_AVAILABLE:
        raise CLIError(
            "Stable-baselines3 is required for RL training. Please install with: "
            "pip install 'stable-baselines3>=2.0.0'"
        )
    
    if not GYMNASIUM_ENV_AVAILABLE:
        raise CLIError(
            "Gymnasium environment support is not available. Please install with: "
            "pip install 'gymnasium>=0.29.0'"
        )


def _create_algorithm_factory() -> Dict[str, Any]:
    """Create algorithm factory mapping for supported RL algorithms."""
    if not SB3_AVAILABLE:
        return {}
    
    return {
        'PPO': PPO,
        'SAC': SAC,
        'TD3': TD3,
        'A2C': A2C,
        'DDPG': DDPG
    }


def _create_vectorized_environment(
    env_config: DictConfig,
    n_envs: int = 4,
    vec_env_type: str = 'dummy',
    frame_cache: Optional[FrameCache] = None
) -> Union[DummyVecEnv, SubprocVecEnv]:
    """
    Create vectorized environment for parallel RL training.
    
    Args:
        env_config: Hydra configuration for environment creation
        n_envs: Number of parallel environments
        vec_env_type: Type of vectorized environment ('dummy' or 'subproc')
        frame_cache: Optional FrameCache instance for performance optimization
        
    Returns:
        Vectorized environment instance
    """
    def make_env():
        """Factory function for creating individual environments."""
        env = create_gymnasium_environment(cfg=env_config, frame_cache=frame_cache)
        env = Monitor(env)  # Add monitoring wrapper
        return env
    
    if vec_env_type == 'subproc':
        return SubprocVecEnv([make_env for _ in range(n_envs)])
    else:
        return DummyVecEnv([make_env for _ in range(n_envs)])


def _setup_training_callbacks(
    output_dir: Path,
    checkpoint_freq: int,
    save_freq: int,
    eval_freq: Optional[int] = None
) -> List[Any]:
    """
    Setup training callbacks for model checkpointing and monitoring.
    
    Args:
        output_dir: Directory for saving checkpoints and logs
        checkpoint_freq: Frequency for saving model checkpoints
        save_freq: Frequency for saving training progress
        eval_freq: Frequency for evaluation (optional)
        
    Returns:
        List of configured callbacks
    """
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="rl_model"
    )
    callbacks.append(checkpoint_callback)
    
    # Progress bar for training monitoring
    if not _CLI_CONFIG.get('quiet', False):
        progress_callback = ProgressBarCallback()
        callbacks.append(progress_callback)
    
    return callbacks


@cli.group()
def train() -> None:
    """
    Reinforcement learning training commands for odor plume navigation.
    
    This command group provides comprehensive RL training capabilities using stable-baselines3
    algorithms with the Gymnasium environment wrapper. Supports algorithm selection,
    vectorized training, automatic checkpointing, performance monitoring, and frame caching
    for optimal training performance.
    
    Available algorithms:
    - PPO (Proximal Policy Optimization) - recommended for most scenarios
    - SAC (Soft Actor-Critic) - for continuous control with exploration
    - TD3 (Twin Delayed DDPG) - for deterministic continuous control
    - A2C (Advantage Actor-Critic) - lightweight policy gradient method
    - DDPG (Deep Deterministic Policy Gradient) - for continuous control
    
    Examples:
        # Train PPO agent with default settings
        plume-nav-sim train algorithm --algorithm PPO
        
        # Train SAC with custom timesteps and vectorized environments
        plume-nav-sim train algorithm --algorithm SAC --total-timesteps 100000 --n-envs 8
        
        # Train with custom configuration and frequent checkpointing
        plume-nav-sim train algorithm --algorithm PPO --config custom_rl.yaml --checkpoint-freq 5000
    """
    pass


@train.command()
@click.option('--algorithm', '-a', required=True,
              type=click.Choice(['PPO', 'SAC', 'TD3', 'A2C', 'DDPG'], case_sensitive=False),
              help='RL algorithm to use for training')
@click.option('--total-timesteps', '-t', type=int, default=50000,
              help='Total number of training timesteps')
@click.option('--n-envs', '-n', type=int, default=4,
              help='Number of parallel environments for vectorized training')
@click.option('--vec-env-type', type=click.Choice(['dummy', 'subproc']), default='dummy',
              help='Type of vectorized environment (dummy for single-process, subproc for multi-process)')
@click.option('--checkpoint-freq', type=int, default=10000,
              help='Frequency for saving model checkpoints (in timesteps)')
@click.option('--learning-rate', type=float, default=None,
              help='Learning rate for the algorithm (uses algorithm default if not specified)')
@click.option('--policy', type=str, default='MultiInputPolicy',
              help='Policy architecture to use (MultiInputPolicy recommended for multi-modal observations)')
@click.option('--verbose', type=int, default=1,
              help='Verbosity level for training output (0=silent, 1=info, 2=debug)')
@click.option('--tensorboard-log', type=click.Path(),
              help='Directory for TensorBoard logging')
@click.option('--eval-freq', type=int, default=None,
              help='Frequency for model evaluation during training')
@click.option('--output-dir', '-o', type=click.Path(), default='rl_training_output',
              help='Output directory for trained models and logs')
@click.option('--frame-cache', type=click.Choice(['none', 'lru', 'all']), default='none',
               callback=lambda ctx, param, value: _validate_frame_cache_mode(value),
               help='Frame caching mode: "none" (no caching), "lru" (LRU eviction with 2GB limit), '
                    '"all" (preload all frames for maximum throughput)')
@click.pass_context
def algorithm(ctx, algorithm: str, total_timesteps: int, n_envs: int, vec_env_type: str,
              checkpoint_freq: int, learning_rate: Optional[float], policy: str,
              verbose: int, tensorboard_log: Optional[str], eval_freq: Optional[int],
               frame_cache: str,
              output_dir: str) -> None:
    """
    Train an RL agent using stable-baselines3 algorithms with the Gymnasium environment.
    
    This command creates a Gymnasium-compliant environment from the Hydra configuration
    and trains a policy using the specified algorithm. Supports vectorized environments
    for improved training efficiency and comprehensive monitoring with checkpoints.
    
    Algorithm Selection Guide:
    - PPO: General-purpose, stable, good for most navigation tasks
    - SAC: Sample-efficient, good exploration, handles stochastic environments
    - TD3: Deterministic control, good for precise navigation tasks
    - A2C: Fast training, lower sample efficiency, good for quick experiments
    - DDPG: Continuous control pioneer, may require careful tuning
    
    Examples:
        # Basic PPO training
        plume-nav-sim train algorithm --algorithm PPO --total-timesteps 100000
        
        # High-performance SAC training with vectorized environments
        plume-nav-sim train algorithm --algorithm SAC --n-envs 8 --vec-env-type subproc
        
        # Training with custom learning rate and TensorBoard logging
        plume-nav-sim train algorithm --algorithm TD3 --learning-rate 0.001 --tensorboard-log ./logs
        
        # Quick experiment with frequent checkpointing
        plume-nav-sim train algorithm --algorithm A2C --total-timesteps 25000 --checkpoint-freq 2500
        
        # Training with frame caching for optimal performance
        plume-nav-sim train algorithm --algorithm PPO --frame-cache lru --n-envs 4
        
        # High-throughput training with full frame preloading
        plume-nav-sim train algorithm --algorithm SAC --frame-cache all --n-envs 8
    """
    start_time = time.time()
    
    
    # Handle frame_cache parameter - use default if not passed due to signature issue
    if 'frame_cache' not in locals():
        frame_cache = 'none'
    try:
        # Validate RL dependencies
        _validate_rl_availability()
        
        # Validate Hydra availability and configuration
        _validate_hydra_availability()
        
        if not HydraConfig.initialized():
            logger.error("Hydra configuration not initialized")
            raise CLIError("RL training requires Hydra configuration initialization")
        
        cfg = HydraConfig.get().cfg
        algorithm_name = algorithm.upper()
        
        logger.info(f"Starting RL training with {algorithm_name} algorithm")
        logger.info(f"Training parameters: {total_timesteps} timesteps, {n_envs} environments")
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create algorithm factory
        algorithm_factory = _create_algorithm_factory()
        if algorithm_name not in algorithm_factory:
            raise CLIError(f"Algorithm {algorithm_name} not supported. Available: {list(algorithm_factory.keys())}")
        
        AlgorithmClass = algorithm_factory[algorithm_name]
        
        
        # Create frame cache if requested for RL training
        frame_cache_instance = None
        if frame_cache != "none":
            try:
                # Extract video path from config for cache initialization
                video_path = None
                if hasattr(cfg, 'environment') and hasattr(cfg.environment, 'video_path'):
                    video_path = cfg.environment.video_path
                elif hasattr(cfg, 'video_plume') and hasattr(cfg.video_plume, 'video_path'):
                    video_path = cfg.video_plume.video_path
                
                frame_cache_instance = _create_frame_cache(frame_cache, video_path=video_path)
                if frame_cache_instance:
                    logger.info(f"Frame cache created in '{frame_cache}' mode for RL training")
            except Exception as e:
                logger.warning(f"Failed to create frame cache for RL training, falling back to direct access: {e}")
                frame_cache_instance = None
        # Create vectorized environment
        logger.info(f"Creating {n_envs} vectorized environments ({vec_env_type} type)")
        try:
            if hasattr(cfg, 'environment'):
                env_config = cfg.environment
            else:
                # Fallback to using the entire config
                env_config = cfg
                logger.warning("No 'environment' section found in config, using entire config")
            
            vec_env = _create_vectorized_environment(
                env_config=env_config,
                n_envs=n_envs,
                vec_env_type=vec_env_type,
                frame_cache=frame_cache_instance
            )
            logger.info("Vectorized environment created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create vectorized environment: {e}")
            raise CLIError(f"Environment creation failed: {e}") from e
        
        # Setup training callbacks
        callbacks = _setup_training_callbacks(
            output_dir=output_path,
            checkpoint_freq=checkpoint_freq,
            save_freq=checkpoint_freq,
            eval_freq=eval_freq
        )
        
        # Prepare algorithm parameters
        algorithm_kwargs = {
            'policy': policy,
            'env': vec_env,
            'verbose': verbose,
            'tensorboard_log': tensorboard_log
        }
        
        # Add learning rate if specified
        if learning_rate is not None:
            algorithm_kwargs['learning_rate'] = learning_rate
        
        # Create and configure the algorithm
        logger.info(f"Initializing {algorithm_name} algorithm with policy '{policy}'")
        model = AlgorithmClass(**algorithm_kwargs)
        
        # Start training with progress monitoring
        logger.info(f"Beginning training for {total_timesteps} timesteps...")
        training_start = time.time()
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name=f"{algorithm_name.lower()}_training",
                reset_num_timesteps=True,
                progress_bar=verbose > 0
            )
            
            training_duration = time.time() - training_start
            logger.info(f"Training completed in {training_duration:.2f}s")
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            training_duration = time.time() - training_start
            logger.info(f"Training ran for {training_duration:.2f}s before interruption")
        
        # Save final model
        final_model_path = output_path / f"final_{algorithm_name.lower()}_model"
        model.save(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")
        
        # Generate training summary
        summary = {
            "algorithm": algorithm_name,
            "total_timesteps": total_timesteps,
            "n_environments": n_envs,
            "training_duration": training_duration,
            "model_path": str(final_model_path),
            "policy": policy,
            "learning_rate": model.learning_rate if hasattr(model, 'learning_rate') else "default"
        }
        
        # Save training metadata
        import json
        metadata_path = output_path / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Training summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # Clean up environment
        vec_env.close()
        
        _measure_performance("RL training", start_time)
        click.echo(click.style(f"✓ RL training completed successfully!", fg='green'))
        click.echo(f"Model saved to: {click.style(str(final_model_path), fg='cyan')}")
        click.echo(f"Training metadata: {click.style(str(metadata_path), fg='cyan')}")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        ctx.exit(1)
    except (CLIError, ConfigValidationError, ConfigurationError) as e:
        logger.error(str(e))
        ctx.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if ctx.obj.get('verbose'):
            logger.error(traceback.format_exc())
        ctx.exit(1)


1006: # Hydra main decorator for configuration injection
1007: @hydra.main(config_path="../../conf", config_name="config", version_base=None)
1008: def main(cfg: DictConfig) -> None:
1009:     """
1010:     Main CLI entrypoint with Hydra configuration management.
1011:     
1012:     This function serves as the primary entry point for the CLI with automatic
1013:     Hydra configuration loading, parameter injection, and multi-run support.
1014:     The @hydra.main decorator enables sophisticated configuration management
1015:     including parameter overrides, configuration composition, and multi-run experiments.
1016:     
1017:     The function processes command-line arguments through Click framework while
1018:     maintaining access to the full Hydra configuration hierarchy for advanced
1019:     parameter management and experiment orchestration.
1020:     
1021:     Args:
1022:         cfg: Hydra configuration object with full hierarchy loaded from conf/ directory
1023:     """
1024:     try:
1025:         # Initialize global timing
1026:         _CLI_CONFIG['start_time'] = time.time()
1027:         
1028:         # Store Hydra config for CLI commands
1029:         if not hasattr(cli, '_hydra_cfg'):
1030:             cli._hydra_cfg = cfg
1031:         
1032:         # Process CLI commands
1033:         cli(standalone_mode=False)
1034:         
1035:     except SystemExit as e:
1036:         # Handle normal CLI exits
1037:         if e.code != 0:
1038:             logger.error(f"CLI exited with code {e.code}")
1039:         sys.exit(e.code)
1040:     except KeyboardInterrupt:
1041:         logger.warning("CLI interrupted by user")
1042:         sys.exit(1)
1043:     except Exception as e:
1044:         logger.error(f"Unexpected CLI error: {e}")
1045:         logger.debug(traceback.format_exc())
1046:         sys.exit(1)
1047: 
1048: 
1049: # Entry point for direct module execution
1050: if __name__ == "__main__":
1051:     main()