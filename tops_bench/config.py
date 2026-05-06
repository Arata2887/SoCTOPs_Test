from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import SUPPORTED_PRECISIONS, SUPPORTED_TASKS


@dataclass(slots=True)
class RuntimeConfig:
    engine: str = "onnxruntime"
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    provider_options: dict[str, dict[str, str]] = field(default_factory=dict)
    intra_op_num_threads: int | None = None
    inter_op_num_threads: int | None = None
    graph_optimization_level: str | None = "ORT_ENABLE_ALL"
    execution_mode: str | None = None
    enable_cpu_mem_arena: bool | None = None
    mock_latency_ms: float = 1.0


@dataclass(slots=True)
class SoCConfig:
    platform: str
    peak_tops: dict[str, float]


@dataclass(slots=True)
class ModelInputSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(slots=True)
class ModelConfig:
    model_id: str
    name: str
    task: str
    ops_per_inference: float
    model_paths: dict[str, Path]
    inputs: list[ModelInputSpec]
    source: str | None = None


@dataclass(slots=True)
class DataConfig:
    random_benchmark: bool = True
    real_data: dict[str, list[Path]] = field(default_factory=dict)
    max_real_samples: int = 4


@dataclass(slots=True)
class BenchmarkConfig:
    warmup_runs: int = 10
    duration_sec: float = 5.0
    repeats: int = 3
    batch_size: int = 1


@dataclass(slots=True)
class AppConfig:
    config_path: Path
    runtime: RuntimeConfig
    soc: SoCConfig
    models: list[ModelConfig]
    data: DataConfig
    benchmark: BenchmarkConfig


def parse_csv_selection(raw: str, *, allowed: tuple[str, ...], label: str) -> list[str]:
    values = [chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError(f"{label} cannot be empty")

    deduped: list[str] = []
    for value in values:
        if value not in allowed:
            allowed_text = ", ".join(allowed)
            raise ValueError(f"Unsupported {label}: {value}. Allowed: {allowed_text}")
        if value not in deduped:
            deduped.append(value)
    return deduped


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    _validate_top_level(raw)

    runtime = _parse_runtime(raw.get("runtime", {}))
    soc = _parse_soc(raw["soc"])
    benchmark = _parse_benchmark(raw.get("benchmark", {}))
    data = _parse_data(raw.get("data", {}), base_dir=path.parent)
    models = _parse_models(raw["models"], base_dir=path.parent)

    app_config = AppConfig(
        config_path=path,
        runtime=runtime,
        soc=soc,
        models=models,
        data=data,
        benchmark=benchmark,
    )
    _validate_semantics(app_config)
    return app_config


def validate_model_paths(
    config: AppConfig,
    precisions: list[str],
    *,
    allow_missing_int8: bool = True,
) -> None:
    for model in config.models:
        for precision in precisions:
            model_path = model.model_paths.get(precision)
            if model_path is None:
                if precision == "int8" and allow_missing_int8:
                    continue
                raise ValueError(
                    f"Model '{model.model_id}' missing model path for precision '{precision}'"
                )
            if not model_path.exists():
                if precision == "int8" and allow_missing_int8:
                    continue
                raise ValueError(
                    f"Model '{model.model_id}' path not found for precision '{precision}': {model_path}"
                )


def _validate_top_level(raw: dict[str, Any]) -> None:
    for key in ("soc", "models"):
        if key not in raw:
            raise ValueError(f"Missing required config section: '{key}'")


def _parse_runtime(raw: dict[str, Any]) -> RuntimeConfig:
    providers = raw.get("providers") or ["CPUExecutionProvider"]
    if not isinstance(providers, list) or not providers:
        raise ValueError("runtime.providers must be a non-empty list")

    return RuntimeConfig(
        engine=str(raw.get("engine", "onnxruntime")).lower(),
        providers=[str(x) for x in providers],
        provider_options={
            str(k): {str(pk): str(pv) for pk, pv in v.items()}
            for k, v in (raw.get("provider_options") or {}).items()
        },
        intra_op_num_threads=_optional_int(raw.get("intra_op_num_threads")),
        inter_op_num_threads=_optional_int(raw.get("inter_op_num_threads")),
        graph_optimization_level=_optional_str(raw.get("graph_optimization_level", "ORT_ENABLE_ALL")),
        execution_mode=_optional_str(raw.get("execution_mode")),
        enable_cpu_mem_arena=_optional_bool(raw.get("enable_cpu_mem_arena")),
        mock_latency_ms=float(raw.get("mock_latency_ms", 1.0)),
    )


def _parse_soc(raw: dict[str, Any]) -> SoCConfig:
    platform = str(raw.get("platform", "unknown"))
    peak_raw = raw.get("peak_tops")
    if not isinstance(peak_raw, dict):
        raise ValueError("soc.peak_tops must be a mapping with fp32/fp16/int8 numeric values")

    peak_tops: dict[str, float] = {}
    for precision, value in peak_raw.items():
        p = str(precision).lower()
        if p not in SUPPORTED_PRECISIONS:
            raise ValueError(f"Unsupported precision under soc.peak_tops: {precision}")
        peak_tops[p] = float(value)
    return SoCConfig(platform=platform, peak_tops=peak_tops)


def _parse_benchmark(raw: dict[str, Any]) -> BenchmarkConfig:
    return BenchmarkConfig(
        warmup_runs=int(raw.get("warmup_runs", 10)),
        duration_sec=float(raw.get("duration_sec", 5.0)),
        repeats=int(raw.get("repeats", 3)),
        batch_size=int(raw.get("batch_size", 1)),
    )


def _parse_data(raw: dict[str, Any], *, base_dir: Path) -> DataConfig:
    real_data_raw = raw.get("real_data") or {}
    real_data: dict[str, list[Path]] = {}
    if not isinstance(real_data_raw, dict):
        raise ValueError("data.real_data must be a mapping from task to path list")

    for task, values in real_data_raw.items():
        task_norm = str(task).lower()
        if task_norm not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task in data.real_data: {task}")
        if not isinstance(values, list):
            raise ValueError(f"data.real_data.{task} must be a list of paths")
        resolved = [_resolve_path(Path(str(v)), base_dir=base_dir) for v in values]
        real_data[task_norm] = resolved

    return DataConfig(
        random_benchmark=bool(raw.get("random_benchmark", True)),
        real_data=real_data,
        max_real_samples=int(raw.get("max_real_samples", 4)),
    )


def _parse_models(raw: Any, *, base_dir: Path) -> list[ModelConfig]:
    if isinstance(raw, dict):
        items = [(str(model_id), body) for model_id, body in raw.items()]
    elif isinstance(raw, list):
        items = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("Each model entry in list form must be a mapping")
            model_id = item.get("id")
            if not model_id:
                raise ValueError("Model entry in list form requires id")
            items.append((str(model_id), item))
    else:
        raise ValueError("models must be either a mapping or a list")

    parsed: list[ModelConfig] = []
    for model_id, body in items:
        if not isinstance(body, dict):
            raise ValueError(f"Model '{model_id}' definition must be a mapping")

        task = str(body.get("task", "")).lower()
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Model '{model_id}' has unsupported task: {task}")

        ops_per_inference = float(body.get("ops_per_inference", 0))
        if ops_per_inference <= 0:
            raise ValueError(f"Model '{model_id}' must define positive ops_per_inference")

        inputs_raw = body.get("inputs")
        if not isinstance(inputs_raw, list) or not inputs_raw:
            raise ValueError(f"Model '{model_id}' must define a non-empty inputs list")

        inputs: list[ModelInputSpec] = []
        for idx, input_spec in enumerate(inputs_raw):
            if not isinstance(input_spec, dict):
                raise ValueError(f"Model '{model_id}' input#{idx} must be a mapping")
            name = str(input_spec.get("name", "")).strip()
            if not name:
                raise ValueError(f"Model '{model_id}' input#{idx} missing 'name'")
            shape_raw = input_spec.get("shape")
            if not isinstance(shape_raw, list) or not shape_raw:
                raise ValueError(f"Model '{model_id}' input '{name}' must define non-empty shape")
            shape = tuple(int(v) for v in shape_raw)
            inputs.append(ModelInputSpec(name=name, shape=shape, dtype=str(input_spec.get("dtype", "float32"))))

        model_paths_raw = body.get("model_paths") or {}
        if not isinstance(model_paths_raw, dict):
            raise ValueError(f"Model '{model_id}' model_paths must be a mapping")

        model_paths: dict[str, Path] = {}
        for precision, path_raw in model_paths_raw.items():
            p = str(precision).lower()
            if p not in SUPPORTED_PRECISIONS:
                raise ValueError(f"Model '{model_id}' has unsupported precision key in model_paths: {precision}")
            model_paths[p] = _resolve_path(Path(str(path_raw)), base_dir=base_dir)

        parsed.append(
            ModelConfig(
                model_id=model_id,
                name=str(body.get("name", model_id)),
                task=task,
                ops_per_inference=ops_per_inference,
                model_paths=model_paths,
                inputs=inputs,
                source=_optional_str(body.get("source")),
            )
        )

    return parsed


def _validate_semantics(config: AppConfig) -> None:
    if config.benchmark.batch_size != 1:
        raise ValueError("benchmark.batch_size must be 1 in v1 benchmark protocol")
    if config.benchmark.warmup_runs < 0:
        raise ValueError("benchmark.warmup_runs must be >= 0")
    if config.benchmark.duration_sec <= 0:
        raise ValueError("benchmark.duration_sec must be > 0")
    if config.benchmark.repeats <= 0:
        raise ValueError("benchmark.repeats must be > 0")
    if config.data.max_real_samples <= 0:
        raise ValueError("data.max_real_samples must be > 0")

    if not config.models:
        raise ValueError("At least one model must be configured")

    # All model IDs must be unique.
    seen: set[str] = set()
    for model in config.models:
        if model.model_id in seen:
            raise ValueError(f"Duplicate model id: {model.model_id}")
        seen.add(model.model_id)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
