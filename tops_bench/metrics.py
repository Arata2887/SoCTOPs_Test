from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LatencyStats:
    avg: float
    p50: float
    p90: float
    p99: float


def compute_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    if not latencies_ms:
        raise ValueError("latencies_ms cannot be empty")

    arr = np.asarray(latencies_ms, dtype=np.float64)
    return LatencyStats(
        avg=float(np.mean(arr)),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p99=float(np.percentile(arr, 99)),
    )


def compute_fps(latencies_ms: list[float]) -> float:
    if not latencies_ms:
        return 0.0
    total_sec = sum(latencies_ms) / 1000.0
    if total_sec <= 0:
        return 0.0
    return len(latencies_ms) / total_sec


def compute_effective_tops(fps: float, ops_per_inference: float) -> float:
    if fps <= 0 or ops_per_inference <= 0:
        return 0.0
    return fps * ops_per_inference / 1e12


def compute_utilization(effective_tops: float, peak_tops: float | None) -> float | None:
    if peak_tops is None or peak_tops <= 0:
        return None
    return effective_tops / peak_tops
