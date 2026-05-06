from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .results import BenchmarkRunResult, ResultRecord


DEFAULT_JSON_NAME = "benchmark_results.json"
DEFAULT_CSV_NAME = "benchmark_results.csv"
DEFAULT_MD_NAME = "benchmark_report.md"


def write_run_outputs(result: BenchmarkRunResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / DEFAULT_JSON_NAME
    csv_path = output_dir / DEFAULT_CSV_NAME
    md_path = output_dir / DEFAULT_MD_NAME

    write_json_result(result, json_path)
    write_csv_result(result, csv_path)
    md_path.write_text(render_markdown(result), encoding="utf-8")

    return {"json": json_path, "csv": csv_path, "md": md_path}


def write_json_result(result: BenchmarkRunResult, path: Path) -> None:
    payload = result.to_dict()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv_result(result: BenchmarkRunResult, path: Path) -> None:
    columns = [
        "model_id",
        "model_name",
        "task",
        "precision",
        "status",
        "latency_avg_ms",
        "latency_p50_ms",
        "latency_p90_ms",
        "latency_p99_ms",
        "fps",
        "effective_tops",
        "peak_tops",
        "utilization",
        "notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in result.records:
            writer.writerow(
                {
                    "model_id": record.model_id,
                    "model_name": record.model_name,
                    "task": record.task,
                    "precision": record.precision,
                    "status": record.status,
                    "latency_avg_ms": _fmt(record.latency_ms.get("avg")),
                    "latency_p50_ms": _fmt(record.latency_ms.get("p50")),
                    "latency_p90_ms": _fmt(record.latency_ms.get("p90")),
                    "latency_p99_ms": _fmt(record.latency_ms.get("p99")),
                    "fps": _fmt(record.fps),
                    "effective_tops": _fmt(record.effective_tops),
                    "peak_tops": _fmt(record.peak_tops),
                    "utilization": _fmt(record.utilization),
                    "notes": " | ".join(record.notes),
                }
            )


def load_run_result(input_path: str | Path) -> BenchmarkRunResult:
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        path = path / DEFAULT_JSON_NAME
    if not path.exists():
        raise ValueError(f"Result json not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    records = [
        ResultRecord(
            model_id=item["model_id"],
            model_name=item["model_name"],
            task=item["task"],
            precision=item["precision"],
            status=item["status"],
            latency_ms=item.get("latency_ms") or {},
            fps=item.get("fps"),
            effective_tops=item.get("effective_tops"),
            peak_tops=item.get("peak_tops"),
            utilization=item.get("utilization"),
            notes=item.get("notes") or [],
        )
        for item in data.get("records", [])
    ]

    return BenchmarkRunResult(
        created_at_utc=data.get("created_at_utc", ""),
        config_path=data.get("config_path", ""),
        platform=data.get("platform", ""),
        runtime_engine=data.get("runtime_engine", ""),
        providers=list(data.get("providers", [])),
        tasks=list(data.get("tasks", [])),
        precisions=list(data.get("precisions", [])),
        records=records,
        notes=list(data.get("notes", [])),
    )


def render_markdown(result: BenchmarkRunResult) -> str:
    lines: list[str] = []
    lines.append("# SoC TOPS Benchmark Report")
    lines.append("")
    lines.append(f"- Platform: `{result.platform}`")
    lines.append(f"- Runtime Engine: `{result.runtime_engine}`")
    lines.append(f"- Providers: `{', '.join(result.providers)}`")
    lines.append(f"- Config: `{result.config_path}`")
    lines.append(f"- Tasks: `{', '.join(result.tasks)}`")
    lines.append(f"- Precisions: `{', '.join(result.precisions)}`")
    lines.append(f"- Created (UTC): `{result.created_at_utc}`")
    if result.notes:
        lines.append(f"- Notes: {'; '.join(result.notes)}")
    lines.append("")

    sorted_records = sorted(
        result.records,
        key=lambda r: (r.utilization is None, -(r.utilization or -1.0), r.task, r.model_id, r.precision),
    )

    lines.append("## Result Table (Sorted by Utilization)")
    lines.append("")
    lines.append(
        "| Task | Model | Precision | Status | Avg Latency (ms) | FPS | Effective TOPS | Peak TOPS | Utilization | Notes |"
    )
    lines.append(
        "|---|---|---|---|---:|---:|---:|---:|---:|---|"
    )

    for r in sorted_records:
        lines.append(
            "| "
            f"{r.task} | {r.model_name} | {r.precision} | {r.status} | "
            f"{_fmt(r.latency_ms.get('avg'))} | {_fmt(r.fps)} | {_fmt(r.effective_tops)} | "
            f"{_fmt(r.peak_tops)} | {_fmt_percent(r.utilization)} | {'; '.join(r.notes)} |"
        )

    lines.append("")
    lines.append("## Status Legend")
    lines.append("")
    lines.append("- `ok`: benchmark completed and metrics are valid")
    lines.append("- `skipped_na`: expected in v1 for missing INT8 artifact")
    lines.append("- `failed`: run attempted but ended with an error")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"
