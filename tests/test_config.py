from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tops_bench.config import load_config, parse_csv_selection, validate_model_paths


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_config(tmp_path: Path) -> dict:
    model_path = tmp_path / "model_fp32.onnx"
    model_path.write_text("dummy", encoding="utf-8")

    return {
        "soc": {
            "platform": "test_soc",
            "peak_tops": {"fp32": 1.0, "fp16": 2.0, "int8": 4.0},
        },
        "models": {
            "resnet18_cls": {
                "name": "resnet18",
                "task": "cls",
                "ops_per_inference": 1_000_000_000,
                "model_paths": {
                    "fp32": str(model_path),
                    "fp16": str(model_path),
                },
                "inputs": [
                    {
                        "name": "input",
                        "shape": [1, 3, 224, 224],
                        "dtype": "float32",
                    }
                ],
            }
        },
        "benchmark": {"warmup_runs": 1, "duration_sec": 0.01, "repeats": 1, "batch_size": 1},
    }


def test_load_config_missing_required_section(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    _write_yaml(config_path, {"models": {}})

    with pytest.raises(ValueError, match="Missing required config section: 'soc'"):
        load_config(config_path)


def test_parse_csv_selection_invalid_precision() -> None:
    with pytest.raises(ValueError, match="Unsupported precisions"):
        parse_csv_selection("fp32,fp64", allowed=("fp32", "fp16", "int8"), label="precisions")


def test_validate_model_paths_detects_missing_non_int8(tmp_path: Path) -> None:
    config_payload = _base_config(tmp_path)
    config_payload["models"]["resnet18_cls"]["model_paths"]["fp32"] = str(tmp_path / "does_not_exist.onnx")
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config_payload)

    config = load_config(config_path)

    with pytest.raises(ValueError, match="path not found"):
        validate_model_paths(config, ["fp32"], allow_missing_int8=True)
