from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .benchmark import run_benchmark
from .config import SUPPORTED_PRECISIONS, SUPPORTED_TASKS, load_config, parse_csv_selection
from .reporting import load_run_result, render_markdown, write_run_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tops_bench",
        description="SoC TOPS benchmark for classification, detection, keypoint, and segmentation models.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run benchmark and write JSON/CSV/Markdown outputs")
    run_parser.add_argument("--config", required=True, help="Path to benchmark YAML config")
    run_parser.add_argument(
        "--tasks",
        default=",".join(SUPPORTED_TASKS),
        help=f"Comma-separated tasks. Allowed: {', '.join(SUPPORTED_TASKS)}",
    )
    run_parser.add_argument(
        "--precisions",
        default=",".join(SUPPORTED_PRECISIONS),
        help=f"Comma-separated precisions. Allowed: {', '.join(SUPPORTED_PRECISIONS)}",
    )
    run_parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base output directory. A timestamped subdirectory will be created.",
    )
    run_parser.add_argument(
        "--name",
        default=None,
        help="Optional output subdirectory name. Defaults to a timestamp.",
    )

    report_parser = subparsers.add_parser("report", help="Render markdown report from a benchmark JSON/result dir")
    report_parser.add_argument("--input", required=True, help="Input result JSON or result directory")
    report_parser.add_argument("--output", required=True, help="Output markdown file path")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_command(args)
    if args.command == "report":
        return _report_command(args)

    parser.print_help()
    return 1


def _run_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    tasks = parse_csv_selection(args.tasks, allowed=SUPPORTED_TASKS, label="tasks")
    precisions = parse_csv_selection(args.precisions, allowed=SUPPORTED_PRECISIONS, label="precisions")

    result = run_benchmark(config=config, tasks=tasks, precisions=precisions)

    output_root = Path(args.output_dir).expanduser().resolve()
    run_name = args.name or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_dir = output_root / run_name

    outputs = write_run_outputs(result, output_dir)

    print(f"Benchmark finished. Outputs written to: {output_dir}")
    print(f"- JSON: {outputs['json']}")
    print(f"- CSV : {outputs['csv']}")
    print(f"- MD  : {outputs['md']}")
    return 0


def _report_command(args: argparse.Namespace) -> int:
    result = load_run_result(args.input)
    markdown = render_markdown(result)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Markdown report generated: {output_path}")
    return 0
