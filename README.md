# SoC TOPS Benchmark

## 1. 项目概述

本项目用于对 SoC 平台进行统一的 AI 推理性能测试，输出模型级别的延迟、吞吐与 TOPS 利用率指标。项目面向对外测试协作场景，适用于将同一套测试流程交付给芯片开发商或平台供应商执行。

当前版本支持以下任务类型：
- 图像分类（`cls`）
- 目标检测（`det`）
- 关键点估计（`kpt`）
- 图像分割（`seg`）

## 2. 测试目标与指标定义

### 2.1 测试目标

- 在统一协议下评估不同精度（`fp32`、`fp16`、`int8`）的推理性能。
- 在相同模型配置下输出可横向对比的性能结果。
- 为 SoC 峰值算力评估提供可复现的数据基础。

### 2.2 输出指标

- `latency_ms`：延迟统计（`avg` / `p50` / `p90` / `p99`）
- `fps`：吞吐（Frames Per Second）
- `effective_tops`：有效 TOPS，计算公式：`FPS * ops_per_inference / 1e12`
- `peak_tops`：配置中的平台标称峰值 TOPS
- `utilization`：算力利用率，计算公式：`effective_tops / peak_tops`
- `status`：执行状态（`ok` / `skipped_na` / `failed`）

说明：当 INT8 模型文件未提供时，INT8 项将标记为 `skipped_na`，不会中断其余测试项。

## 3. 环境与依赖管理（uv）

本项目采用 `uv` 作为唯一推荐环境管理方案，用于保证对外共享时的依赖一致性与可复现性。

### 3.1 环境要求

- Python：`3.11`（仓库包含 `.python-version`）
- 工具：`uv`

### 3.2 依赖同步

项目默认依赖组为 `runtime + dev`，执行以下命令即可完成标准环境安装：

```bash
uv sync
```

如仅需最小运行依赖：

```bash
uv sync --no-default-groups
```

## 4. 使用流程

### 4.1 执行基准测试

```bash
uv run python -m tops_bench run \
  --config configs/benchmark.sample.yaml \
  --tasks cls,det,kpt,seg \
  --precisions fp32,fp16,int8 \
  --output-dir outputs
```

### 4.2 生成报告

```bash
uv run python -m tops_bench report \
  --input outputs/<run_dir> \
  --output outputs/<run_dir>/benchmark_report.md
```

### 4.3 常用检查命令

```bash
uv run pytest
uv run python -m tops_bench --help
uv run python -m tops_bench run --help
uv run python -m tops_bench report --help
```

## 5. 配置说明

配置文件参考：`configs/benchmark.sample.yaml`

关键字段说明：
- `runtime`：推理后端配置（默认 `onnxruntime`）
- `soc.peak_tops`：平台在不同精度下的峰值 TOPS
- `models[*].ops_per_inference`：单次推理操作量（用于有效 TOPS 计算）
- `models[*].model_paths`：各精度模型文件路径
- `benchmark.batch_size`：当前版本固定为 `1`

## 6. 输入数据约定

`data.real_data` 支持按任务配置 `.npy` / `.npz` 文件或目录：
- `.npy`：写入模型第一个输入
- `.npz`：按输入名匹配

若未提供真实样本，系统将使用随机输入完成性能压测，并在报告中记录说明。

## 7. 输出产物

每次 `run` 执行后将生成：
- `benchmark_results.json`
- `benchmark_results.csv`
- `benchmark_report.md`

其中 Markdown 报告默认按利用率排序，便于快速对比不同模型与精度组合。

## 8. 适用范围与限制

- 本项目聚焦性能基准流程，不包含 INT8 量化产物生成（默认接收外部提供的 INT8 模型）。
- 在跨硬件平台对比时，应确保模型文件、输入尺寸、`ops_per_inference` 与计时参数保持一致。
- 如需对接厂商专用执行后端，可通过配置切换 runtime/provider 实现扩展。

## 9. 默认模型清单（已落地到 `models/`）

- 分类：`resnet18_fp32.onnx` / `resnet18_fp16.onnx`
- 检测：`fasterrcnn_mbv3_320_fp32.onnx` / `fasterrcnn_mbv3_320_fp16.onnx`
- 关键点：`keypointrcnn_resnet50_fp32.onnx` / `keypointrcnn_resnet50_fp16.onnx`
- 分割：`fcn_resnet50_fp32.onnx` / `fcn_resnet50_fp16.onnx`

如需重新导出模型，可执行：

```bash
uv run --with torch --with torchvision --with onnx --with onnxconverter-common --with onnxscript \
  python scripts/export_models_torchvision.py
```
