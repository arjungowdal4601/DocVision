from pathlib import Path

from vision_indexer.schemas.run_status import RunStatus
from vision_indexer.storage.run_status_store import (
    initialize_run_status,
    load_run_status,
    mark_run_completed,
    mark_run_failed,
    mark_run_resuming,
    save_run_status,
    update_page_status,
)


def test_run_status_store_initializes_saves_loads_and_updates_pages(tmp_path: Path) -> None:
    status = initialize_run_status(
        run_id="status_run",
        pdf_path="input.pdf",
        output_dir=str(tmp_path),
        page_paths=["page_0001.png", "page_0002.png"],
    )

    status = update_page_status(status, page_number=1, status="running", attempts=1)
    status = update_page_status(
        status,
        page_number=1,
        status="completed",
        output_path="page_outputs/page_0001.json",
    )
    save_run_status(tmp_path, status)

    loaded = load_run_status(tmp_path)

    assert isinstance(loaded, RunStatus)
    assert loaded.page_statuses["1"].status == "completed"
    assert loaded.completed_pages == [1]
    assert loaded.failed_pages == []


def test_run_status_store_marks_run_lifecycle_states(tmp_path: Path) -> None:
    status = initialize_run_status(
        run_id="status_run",
        pdf_path="input.pdf",
        output_dir=str(tmp_path),
        page_paths=["page_0001.png"],
    )

    resuming = mark_run_resuming(status)
    failed = mark_run_failed(resuming, current_page=1)
    completed = mark_run_completed(failed)

    assert resuming.status == "resuming"
    assert failed.status == "failed"
    assert failed.current_page == 1
    assert completed.status == "completed"
    assert completed.finished_at is not None
