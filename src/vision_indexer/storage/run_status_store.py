from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from vision_indexer.schemas.run_status import PageStatus, PageStatusValue, RunStatus


RUN_STATUS_FILENAME = "run_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_run_status(output_dir: Path) -> RunStatus | None:
    status_path = output_dir / RUN_STATUS_FILENAME
    if not status_path.exists():
        return None
    return RunStatus.model_validate_json(status_path.read_text(encoding="utf-8"))


def save_run_status(output_dir: Path, status: RunStatus) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RUN_STATUS_FILENAME).write_text(status.model_dump_json(indent=2), encoding="utf-8")


def initialize_run_status(
    run_id: str,
    pdf_path: str,
    output_dir: str,
    page_paths: list[str],
) -> RunStatus:
    now = utc_now()
    page_statuses = {
        str(index): PageStatus(page_number=index, page_path=page_path)
        for index, page_path in enumerate(page_paths, start=1)
    }
    return RunStatus(
        run_id=run_id,
        pdf_path=pdf_path,
        output_dir=output_dir,
        status="running",
        total_pages=len(page_paths),
        started_at=now,
        page_statuses=page_statuses,
    )


def update_page_status(
    run_status: RunStatus,
    page_number: int,
    status: PageStatusValue,
    page_path: str | None = None,
    error_message: str | None = None,
    attempts: int | None = None,
    output_path: str | None = None,
) -> RunStatus:
    updated = run_status.model_copy(deep=True)
    key = str(page_number)
    existing = updated.page_statuses.get(
        key,
        PageStatus(page_number=page_number, page_path=page_path or ""),
    )
    now = utc_now()

    page_status = existing.model_copy(
        update={
            "page_path": page_path or existing.page_path,
            "status": status,
            "error_message": error_message,
            "attempts": existing.attempts if attempts is None else attempts,
            "output_path": output_path if output_path is not None else existing.output_path,
        }
    )
    if status == "running":
        page_status.started_at = now
        page_status.finished_at = None
    elif status in {"completed", "failed", "skipped"}:
        page_status.finished_at = now

    updated.page_statuses[key] = page_status
    updated.current_page = page_number if status in {"running", "failed"} else updated.current_page

    if status == "completed":
        updated.completed_pages = sorted({*updated.completed_pages, page_number})
        updated.failed_pages = [page for page in updated.failed_pages if page != page_number]
    elif status == "failed":
        updated.failed_pages = sorted({*updated.failed_pages, page_number})
        updated.completed_pages = [page for page in updated.completed_pages if page != page_number]

    return updated


def mark_run_completed(run_status: RunStatus) -> RunStatus:
    return run_status.model_copy(
        update={
            "status": "completed",
            "current_page": None,
            "finished_at": utc_now(),
            "failed_pages": [],
        },
        deep=True,
    )


def mark_run_failed(run_status: RunStatus, current_page: int | None) -> RunStatus:
    return run_status.model_copy(
        update={
            "status": "failed",
            "current_page": current_page,
            "finished_at": utc_now(),
        },
        deep=True,
    )


def mark_run_resuming(run_status: RunStatus) -> RunStatus:
    return run_status.model_copy(
        update={
            "status": "resuming",
            "current_page": None,
            "finished_at": None,
            "resume_count": run_status.resume_count + 1,
        },
        deep=True,
    )


def mark_run_running(run_status: RunStatus) -> RunStatus:
    return run_status.model_copy(update={"status": "running", "finished_at": None}, deep=True)
