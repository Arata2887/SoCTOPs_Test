from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import RuntimeConfig


class InferenceRunner(Protocol):
    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        ...


@dataclass(slots=True)
class ONNXRuntimeRunner:
    runtime: RuntimeConfig
    model_path: Path
    _session: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for runtime.engine=onnxruntime. "
                "Install it with: pip install onnxruntime"
            ) from exc

        sess_options = ort.SessionOptions()
        if self.runtime.intra_op_num_threads is not None:
            sess_options.intra_op_num_threads = self.runtime.intra_op_num_threads
        if self.runtime.inter_op_num_threads is not None:
            sess_options.inter_op_num_threads = self.runtime.inter_op_num_threads
        if self.runtime.enable_cpu_mem_arena is not None:
            sess_options.enable_cpu_mem_arena = self.runtime.enable_cpu_mem_arena

        if self.runtime.graph_optimization_level:
            level_name = self.runtime.graph_optimization_level
            if not hasattr(ort.GraphOptimizationLevel, level_name):
                raise ValueError(f"Unknown graph_optimization_level: {level_name}")
            sess_options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level_name)

        if self.runtime.execution_mode:
            mode_name = self.runtime.execution_mode
            if not hasattr(ort.ExecutionMode, mode_name):
                raise ValueError(f"Unknown execution_mode: {mode_name}")
            sess_options.execution_mode = getattr(ort.ExecutionMode, mode_name)

        provider_options = []
        if self.runtime.provider_options:
            for provider in self.runtime.providers:
                provider_options.append(self.runtime.provider_options.get(provider, {}))

        kwargs: dict[str, Any] = {
            "sess_options": sess_options,
            "providers": self.runtime.providers,
        }
        if provider_options:
            kwargs["provider_options"] = provider_options

        self._session = ort.InferenceSession(str(self.model_path), **kwargs)

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        outputs = self._session.run(None, inputs)
        return [np.asarray(x) for x in outputs]


@dataclass(slots=True)
class MockRunner:
    runtime: RuntimeConfig
    model_path: Path

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self.runtime.mock_latency_ms > 0:
            time.sleep(self.runtime.mock_latency_ms / 1000.0)
        # Keep behavior deterministic enough for tests.
        total = 0.0
        for arr in inputs.values():
            total += float(np.mean(arr))
        return [np.array([total], dtype=np.float32)]


def create_runner(runtime: RuntimeConfig, model_path: Path) -> InferenceRunner:
    if runtime.engine == "onnxruntime":
        return ONNXRuntimeRunner(runtime=runtime, model_path=model_path)
    if runtime.engine == "mock":
        return MockRunner(runtime=runtime, model_path=model_path)
    raise ValueError(f"Unsupported runtime.engine: {runtime.engine}")
