from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import STATUS_FAILED, STATUS_OK, STATUS_SKIPPED_NA


@dataclass(slots=True)
class ResultRecord:
    model_id: str
    model_name: str
    task: str
    precision: str
    status: str
    latency_ms: dict[str, float | None]
    fps: float | None
    effective_tops: float | None
    peak_tops: float | None
    utilization: float | None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def skipped_na(
        cls,
        *,
        model_id: str,
        model_name: str,
        task: str,
        precision: str,
        note: str,
        peak_tops: float | None,
    ) -> "ResultRecord":
        return cls(
            model_id=model_id,
            model_name=model_name,
            task=task,
            precision=precision,
            status=STATUS_SKIPPED_NA,
            latency_ms={"avg": None, "p50": None, "p90": None, "p99": None},
            fps=None,
            effective_tops=None,
            peak_tops=peak_tops,
            utilization=None,
            notes=[note],
        )

    @classmethod
    def failed(
        cls,
        *,
        model_id: str,
        model_name: str,
        task: str,
        precision: str,
        note: str,
        peak_tops: float | None,
    ) -> "ResultRecord":
        return cls(
            model_id=model_id,
            model_name=model_name,
            task=task,
            precision=precision,
            status=STATUS_FAILED,
            latency_ms={"avg": None, "p50": None, "p90": None, "p99": None},
            fps=None,
            effective_tops=None,
            peak_tops=peak_tops,
            utilization=None,
            notes=[note],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkRunResult:
    created_at_utc: str
    config_path: str
    platform: str
    runtime_engine: str
    providers: list[str]
    tasks: list[str]
    precisions: list[str]
    records: list[ResultRecord]
    notes: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        config_path: Path,
        platform: str,
        runtime_engine: str,
        providers: list[str],
        tasks: list[str],
        precisions: list[str],
    ) -> "BenchmarkRunResult":
        return cls(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            config_path=str(config_path),
            platform=platform,
            runtime_engine=runtime_engine,
            providers=providers,
            tasks=tasks,
            precisions=precisions,
            records=[],
            notes=[],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [record.to_dict() for record in self.records]
        return payload


def classify_record(record: ResultRecord) -> str:
    if record.status == STATUS_OK:
        return "ok"
    if record.status == STATUS_SKIPPED_NA:
        return "skipped_na"
    return "failed"
