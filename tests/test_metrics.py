from __future__ import annotations

from tops_bench.metrics import (
    compute_effective_tops,
    compute_fps,
    compute_latency_stats,
    compute_utilization,
)


def test_metrics_formula() -> None:
    latencies_ms = [10.0, 20.0, 30.0]
    stats = compute_latency_stats(latencies_ms)
    fps = compute_fps(latencies_ms)
    effective_tops = compute_effective_tops(fps=fps, ops_per_inference=4_000_000_000)
    utilization = compute_utilization(effective_tops=effective_tops, peak_tops=16.0)

    assert round(stats.avg, 6) == 20.0
    assert round(stats.p50, 6) == 20.0
    assert round(fps, 6) == 50.0
    assert round(effective_tops, 6) == 0.2
    assert round(utilization or 0.0, 6) == 0.0125
