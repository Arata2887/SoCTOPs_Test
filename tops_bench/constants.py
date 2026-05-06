from __future__ import annotations

SUPPORTED_TASKS = ("cls", "det", "kpt", "seg")
SUPPORTED_PRECISIONS = ("fp32", "fp16", "int8")

STATUS_OK = "ok"
STATUS_SKIPPED_NA = "skipped_na"
STATUS_FAILED = "failed"
