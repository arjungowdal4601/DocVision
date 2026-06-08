from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


CSV_CHECKPOINT_FILENAME = "page_checkpoints.csv"
CSV_CHECKPOINT_FIELDS = [
    "timestamp",
    "event",
    "run_id",
    "page_number",
    "status",
    "attempts",
    "output_path",
    "error_path",
    "message",
]


def append_checkpoint_event(
    output_dir: Path,
    *,
    event: str,
    run_id: str,
    page_number: int | None = None,
    status: str | None = None,
    attempts: int | None = None,
    output_path: str | None = None,
    error_path: str | None = None,
    message: str | None = None,
) -> Path:
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / CSV_CHECKPOINT_FILENAME
    should_write_header = not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "run_id": run_id,
        "page_number": "" if page_number is None else str(page_number),
        "status": status or "",
        "attempts": "" if attempts is None else str(attempts),
        "output_path": output_path or "",
        "error_path": error_path or "",
        "message": message or "",
    }

    with checkpoint_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_CHECKPOINT_FIELDS)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)

    return checkpoint_path
