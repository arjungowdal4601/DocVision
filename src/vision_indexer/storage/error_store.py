from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)


def write_page_error(output_dir: Path, page_number: int, error: Exception, attempt: int | None = None) -> Path:
    errors_dir = output_dir / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "page_number": page_number,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "attempt": attempt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    error_path = errors_dir / f"page_{page_number:04d}_error.json"
    error_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.error(
        "Page %s failed on attempt %s with %s: %s",
        page_number,
        attempt,
        payload["error_type"],
        payload["error_message"],
    )
    return error_path


def write_topic_index_error(
    output_dir: Path,
    error: Exception,
    batch_number: int | None = None,
    page_range: str | None = None,
) -> Path:
    errors_dir = output_dir / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "topic_index",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "batch_number": batch_number,
        "page_range": page_range,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    error_path = errors_dir / "topic_index_error.json"
    error_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.error(
        "Topic index failed with %s: %s",
        payload["error_type"],
        payload["error_message"],
    )
    return error_path
