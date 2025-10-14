from benchmarks.environment_performance import (
    EnvironmentBenchmarkConfig,
    PerformanceAnalysis,
    benchmark_rendering_performance,
)
from plume_nav_sim.envs.plume_search_env import create_plume_search_env


def test_benchmark_rendering_performance_contract():
    env = create_plume_search_env()
    try:
        result = benchmark_rendering_performance(env, num_renders=3)
    finally:
        env.close()

    # Contract: result must include basic timing statistics
    assert isinstance(result, dict)
    for key in ("count", "mean_ms", "p95_ms", "std_ms", "format_compliant_rate"):
        assert key in result


def test_performance_analysis_contract():
    analysis = PerformanceAnalysis()

    trends = analysis.analyze_performance_trends(
        {
            "step_latency": {
                "mean_step_time_ms": 1.0,
                "p95_step_time_ms": 2.0,
                "std_dev_step_time_ms": 0.2,
            },
            "memory_usage": {"mean_memory_delta_mb": 0.1, "peak_memory_mb": 10.0},
            "scalability": {"scaling_measurements": {}},
        }
    )
    assert isinstance(trends, dict)

    cfg = EnvironmentBenchmarkConfig()

    class DummyResult:
        config = cfg
        step_latency_metrics = {"mean_step_time_ms": 2.0}
        memory_usage_metrics = {"peak_memory_mb": 1.0}

    recs = analysis.generate_optimization_recommendations(
        DummyResult(), trends, include_scaling_guidance=True
    )
    assert isinstance(recs, list)
