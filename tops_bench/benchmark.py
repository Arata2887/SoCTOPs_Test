from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AppConfig, ModelConfig
from .constants import STATUS_OK
from .metrics import (
    compute_effective_tops,
    compute_fps,
    compute_latency_stats,
    compute_utilization,
)
from .results import BenchmarkRunResult, ResultRecord
from .runner import create_runner


@dataclass(slots=True)
class InputBundle:
    values: dict[str, np.ndarray]


def run_benchmark(
    config: AppConfig,
    *,
    tasks: list[str],
    precisions: list[str],
) -> BenchmarkRunResult:
    result = BenchmarkRunResult.create(
        config_path=config.config_path,
        platform=config.soc.platform,
        runtime_engine=config.runtime.engine,
        providers=config.runtime.providers,
        tasks=tasks,
        precisions=precisions,
    )

    if config.data.random_benchmark:
        result.notes.append("Random input benchmark enabled.")
    else:
        result.notes.append("Random input benchmark disabled; using real data if available.")

    for model in config.models:
        if model.task not in tasks:
            continue
        for precision in precisions:
            record = _benchmark_single(config=config, model=model, precision=precision)
            result.records.append(record)

    return result


def _benchmark_single(config: AppConfig, *, model: ModelConfig, precision: str) -> ResultRecord:
    peak_tops = config.soc.peak_tops.get(precision)

    model_path = model.model_paths.get(precision)
    if model_path is None:
        if precision == "int8":
            return ResultRecord.skipped_na(
                model_id=model.model_id,
                model_name=model.name,
                task=model.task,
                precision=precision,
                note="INT8 model path is not configured.",
                peak_tops=peak_tops,
            )
        return ResultRecord.failed(
            model_id=model.model_id,
            model_name=model.name,
            task=model.task,
            precision=precision,
            note=f"Model path for precision '{precision}' is not configured.",
            peak_tops=peak_tops,
        )

    if not model_path.exists():
        if precision == "int8":
            return ResultRecord.skipped_na(
                model_id=model.model_id,
                model_name=model.name,
                task=model.task,
                precision=precision,
                note=f"INT8 model file not found: {model_path}",
                peak_tops=peak_tops,
            )
        return ResultRecord.failed(
            model_id=model.model_id,
            model_name=model.name,
            task=model.task,
            precision=precision,
            note=f"Model file not found: {model_path}",
            peak_tops=peak_tops,
        )

    notes: list[str] = []
    try:
        runner = create_runner(config.runtime, model_path)
    except Exception as exc:
        return ResultRecord.failed(
            model_id=model.model_id,
            model_name=model.name,
            task=model.task,
            precision=precision,
            note=f"Failed to initialize runner: {exc}",
            peak_tops=peak_tops,
        )

    rng = np.random.default_rng(seed=42)

    real_bundles = _load_real_input_bundles(config=config, model=model, rng=rng)
    if real_bundles:
        # A light correctness/stability pass on real samples.
        try:
            for bundle in real_bundles:
                runner.run(bundle.values)
            notes.append(f"Validated with {len(real_bundles)} real sample(s).")
        except Exception as exc:
            return ResultRecord.failed(
                model_id=model.model_id,
                model_name=model.name,
                task=model.task,
                precision=precision,
                note=f"Real data validation failed: {exc}",
                peak_tops=peak_tops,
            )
    else:
        notes.append("No real samples loaded; benchmark used random input only.")

    def build_perf_input() -> dict[str, np.ndarray]:
        if config.data.random_benchmark or not real_bundles:
            return _build_random_input(model=model, rng=rng)
        return next(real_cycle).values

    real_cycle = itertools.cycle(real_bundles) if real_bundles else None

    try:
        for _ in range(config.benchmark.warmup_runs):
            runner.run(build_perf_input())

        latencies_ms: list[float] = []
        for _ in range(config.benchmark.repeats):
            end_at = time.perf_counter() + config.benchmark.duration_sec
            while time.perf_counter() < end_at:
                inference_input = build_perf_input()
                start = time.perf_counter()
                runner.run(inference_input)
                stop = time.perf_counter()
                latencies_ms.append((stop - start) * 1000.0)

        if not latencies_ms:
            return ResultRecord.failed(
                model_id=model.model_id,
                model_name=model.name,
                task=model.task,
                precision=precision,
                note="No latency samples were captured during benchmark.",
                peak_tops=peak_tops,
            )

        latency = compute_latency_stats(latencies_ms)
        fps = compute_fps(latencies_ms)
        effective_tops = compute_effective_tops(fps=fps, ops_per_inference=model.ops_per_inference)
        utilization = compute_utilization(effective_tops=effective_tops, peak_tops=peak_tops)

        return ResultRecord(
            model_id=model.model_id,
            model_name=model.name,
            task=model.task,
            precision=precision,
            status=STATUS_OK,
            latency_ms={
                "avg": latency.avg,
                "p50": latency.p50,
                "p90": latency.p90,
                "p99": latency.p99,
            },
            fps=fps,
            effective_tops=effective_tops,
            peak_tops=peak_tops,
            utilization=utilization,
            notes=notes,
        )
    except Exception as exc:
        return ResultRecord.failed(
            model_id=model.model_id,
            model_name=model.name,
            task=model.task,
            precision=precision,
            note=f"Benchmark failed: {exc}",
            peak_tops=peak_tops,
        )


def _load_real_input_bundles(
    config: AppConfig,
    *,
    model: ModelConfig,
    rng: np.random.Generator,
) -> list[InputBundle]:
    task_paths = config.data.real_data.get(model.task, [])
    if not task_paths:
        return []

    file_paths = _expand_sample_files(task_paths)
    bundles: list[InputBundle] = []

    for sample_path in file_paths:
        if len(bundles) >= config.data.max_real_samples:
            break
        try:
            bundle = _build_input_from_sample(model=model, sample_path=sample_path, rng=rng)
            bundles.append(bundle)
        except Exception:
            # Ignore malformed sample files in v1; benchmark can still proceed.
            continue

    return bundles


def _expand_sample_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(path.glob("*.npy")))
            files.extend(sorted(path.glob("*.npz")))
    return files


def _build_input_from_sample(
    *,
    model: ModelConfig,
    sample_path: Path,
    rng: np.random.Generator,
) -> InputBundle:
    sample = np.load(sample_path, allow_pickle=False)
    input_map = _build_random_input(model=model, rng=rng)

    if isinstance(sample, np.ndarray):
        first = model.inputs[0]
        input_map[first.name] = _coerce_array(sample, first.shape, first.dtype)
        return InputBundle(values=input_map)

    if isinstance(sample, np.lib.npyio.NpzFile):
        for spec in model.inputs:
            if spec.name in sample:
                input_map[spec.name] = _coerce_array(sample[spec.name], spec.shape, spec.dtype)
        return InputBundle(values=input_map)

    raise ValueError(f"Unsupported sample type from file: {sample_path}")


def _build_random_input(*, model: ModelConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for spec in model.inputs:
        shape = tuple(_normalize_dim(dim) for dim in spec.shape)
        values[spec.name] = _random_array(shape=shape, dtype=spec.dtype, rng=rng)
    return values


def _random_array(*, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
    np_dtype = np.dtype(dtype)
    kind = np_dtype.kind

    if kind in ("f", "c"):
        arr = rng.uniform(-1.0, 1.0, size=shape)
        return arr.astype(np_dtype)
    if kind in ("i", "u"):
        arr = rng.integers(low=0, high=255, size=shape)
        return arr.astype(np_dtype)
    if kind == "b":
        arr = rng.integers(low=0, high=2, size=shape)
        return arr.astype(np_dtype)

    # Fallback to float for unsupported dtypes in v1.
    arr = rng.uniform(-1.0, 1.0, size=shape)
    return arr.astype(np.float32)


def _coerce_array(arr: np.ndarray, expected_shape: tuple[int, ...], expected_dtype: str) -> np.ndarray:
    target_dtype = np.dtype(expected_dtype)
    normalized_shape = tuple(_normalize_dim(dim) for dim in expected_shape)

    out = np.asarray(arr)
    if normalized_shape and all(dim > 0 for dim in normalized_shape):
        if out.shape != normalized_shape:
            expected_count = math.prod(normalized_shape)
            if out.size != expected_count:
                raise ValueError(
                    f"Sample shape mismatch; expected {normalized_shape}, got {out.shape}"
                )
            out = out.reshape(normalized_shape)
    return out.astype(target_dtype)


def _normalize_dim(dim: int) -> int:
    if dim <= 0:
        return 1
    return dim
