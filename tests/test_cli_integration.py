from __future__ import annotations

from pathlib import Path

import yaml

from tops_bench.cli import main
from tops_bench.reporting import DEFAULT_CSV_NAME, DEFAULT_JSON_NAME, DEFAULT_MD_NAME


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_run_and_report_commands_generate_artifacts(tmp_path: Path) -> None:
    model_file = tmp_path / "model.onnx"
    model_file.write_text("dummy", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    payload = {
        "runtime": {"engine": "mock", "mock_latency_ms": 0.0},
        "soc": {
            "platform": "soc_demo",
            "peak_tops": {"fp32": 10.0, "fp16": 20.0, "int8": 40.0},
        },
        "models": {
            "resnet18_cls": {
                "name": "resnet18",
                "task": "cls",
                "ops_per_inference": 1_000_000_000,
                "model_paths": {"fp32": str(model_file)},
                "inputs": [{"name": "input", "shape": [1, 3, 8, 8], "dtype": "float32"}],
            }
        },
        "benchmark": {"warmup_runs": 0, "duration_sec": 0.02, "repeats": 1, "batch_size": 1},
    }
    _write_yaml(config_path, payload)

    output_root = tmp_path / "outputs"
    rc = main(
        [
            "run",
            "--config",
            str(config_path),
            "--tasks",
            "cls",
            "--precisions",
            "fp32,int8",
            "--output-dir",
            str(output_root),
            "--name",
            "test_run",
        ]
    )
    assert rc == 0

    result_dir = output_root / "test_run"
    assert (result_dir / DEFAULT_JSON_NAME).exists()
    assert (result_dir / DEFAULT_CSV_NAME).exists()
    assert (result_dir / DEFAULT_MD_NAME).exists()

    custom_md = result_dir / "regenerated.md"
    rc2 = main(["report", "--input", str(result_dir), "--output", str(custom_md)])
    assert rc2 == 0
    assert custom_md.exists()
