from __future__ import annotations

from pathlib import Path

import yaml

from tops_bench.benchmark import run_benchmark
from tops_bench.config import load_config


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_int8_missing_is_skipped_na(tmp_path: Path) -> None:
    model_file = tmp_path / "model.onnx"
    model_file.write_text("dummy", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    payload = {
        "runtime": {"engine": "mock", "mock_latency_ms": 0.0},
        "soc": {
            "platform": "soc_x",
            "peak_tops": {"fp32": 1.0, "fp16": 2.0, "int8": 4.0},
        },
        "models": {
            "resnet18_cls": {
                "name": "resnet18",
                "task": "cls",
                "ops_per_inference": 1_000_000_000,
                "model_paths": {"fp32": str(model_file), "fp16": str(model_file)},
                "inputs": [{"name": "input", "shape": [1, 3, 8, 8], "dtype": "float32"}],
            }
        },
        "benchmark": {"warmup_runs": 0, "duration_sec": 0.02, "repeats": 1, "batch_size": 1},
    }
    _write_yaml(config_path, payload)

    config = load_config(config_path)
    result = run_benchmark(config, tasks=["cls"], precisions=["fp32", "int8"])

    by_precision = {record.precision: record for record in result.records}
    assert by_precision["fp32"].status == "ok"
    assert by_precision["int8"].status == "skipped_na"
    assert by_precision["int8"].fps is None
