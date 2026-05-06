from __future__ import annotations

from pathlib import Path

import torch
import onnx
from onnxconverter_common import float16
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OPSET = 17


def _export(*, model, args, out: Path, input_names: list[str], output_names: list[str] | None = None) -> None:
    kwargs = {
        "export_params": True,
        "opset_version": OPSET,
        "do_constant_folding": True,
        "input_names": input_names,
        "dynamo": False,  # Use legacy exporter for better torchvision detection compatibility.
    }
    if output_names:
        kwargs["output_names"] = output_names
    torch.onnx.export(model, args, out, **kwargs)


def export_cls() -> None:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    dummy = torch.randn(1, 3, 224, 224)
    out = MODELS_DIR / "resnet18_fp32.onnx"
    _export(model=model, args=dummy, out=out, input_names=["input"], output_names=["logits"])


def export_det() -> None:
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    ).eval()
    images = [torch.randn(3, 320, 320)]
    out = MODELS_DIR / "fasterrcnn_mbv3_320_fp32.onnx"
    _export(model=model, args=(images,), out=out, input_names=["images"])


def export_kpt() -> None:
    model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    ).eval()
    images = [torch.randn(3, 640, 640)]
    out = MODELS_DIR / "keypointrcnn_resnet50_fp32.onnx"
    _export(model=model, args=(images,), out=out, input_names=["images"])


def export_seg() -> None:
    model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1).eval()
    dummy = torch.randn(1, 3, 520, 520)
    out = MODELS_DIR / "fcn_resnet50_fp32.onnx"
    _export(
        model=model,
        args=dummy,
        out=out,
        input_names=["input"],
        output_names=["segmentation"],
    )


def make_fp16(fp32_path: Path, fp16_path: Path) -> None:
    model = onnx.load(fp32_path)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
    )
    onnx.save(model_fp16, fp16_path)


def main() -> None:
    export_cls()
    export_det()
    export_kpt()
    export_seg()

    mapping = {
        "resnet18_fp32.onnx": "resnet18_fp16.onnx",
        "fasterrcnn_mbv3_320_fp32.onnx": "fasterrcnn_mbv3_320_fp16.onnx",
        "keypointrcnn_resnet50_fp32.onnx": "keypointrcnn_resnet50_fp16.onnx",
        "fcn_resnet50_fp32.onnx": "fcn_resnet50_fp16.onnx",
    }
    for src, dst in mapping.items():
        make_fp16(MODELS_DIR / src, MODELS_DIR / dst)

    print("Export complete. Files:")
    for p in sorted(MODELS_DIR.glob("*.onnx")):
        print("-", p.name, f"({p.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
