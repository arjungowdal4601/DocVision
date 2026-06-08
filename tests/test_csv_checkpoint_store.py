from __future__ import annotations

import csv
from pathlib import Path

from vision_indexer.storage.csv_checkpoint_store import CSV_CHECKPOINT_FIELDS, append_checkpoint_event


def _read_rows(run_dir: Path) -> tuple[list[str] | None, list[dict[str, str]]]:
    checkpoint_path = run_dir / "checkpoints" / "page_checkpoints.csv"
    with checkpoint_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_checkpoint_csv_is_created_with_expected_header(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    checkpoint_path = append_checkpoint_event(
        run_dir,
        event="run_started",
        run_id="run",
        message="fresh run",
    )

    fieldnames, rows = _read_rows(run_dir)
    assert checkpoint_path == run_dir / "checkpoints" / "page_checkpoints.csv"
    assert fieldnames == CSV_CHECKPOINT_FIELDS
    assert rows[0]["event"] == "run_started"
    assert rows[0]["run_id"] == "run"
    assert rows[0]["page_number"] == ""
    assert rows[0]["message"] == "fresh run"


def test_checkpoint_csv_appends_events_without_overwriting(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    append_checkpoint_event(run_dir, event="run_started", run_id="run")
    append_checkpoint_event(
        run_dir,
        event="page_started",
        run_id="run",
        page_number=1,
        status="running",
        attempts=1,
    )

    _, rows = _read_rows(run_dir)
    assert [row["event"] for row in rows] == ["run_started", "page_started"]
    assert rows[1]["page_number"] == "1"
    assert rows[1]["status"] == "running"
    assert rows[1]["attempts"] == "1"


def test_resume_checkpoint_appends_to_existing_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    append_checkpoint_event(run_dir, event="run_started", run_id="run")
    append_checkpoint_event(run_dir, event="run_resumed", run_id="run", message="resume count 1")

    _, rows = _read_rows(run_dir)
    assert [row["event"] for row in rows] == ["run_started", "run_resumed"]
    assert rows[1]["message"] == "resume count 1"


def test_checkpoint_store_does_not_create_sqlite_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    append_checkpoint_event(run_dir, event="run_started", run_id="run")

    assert not list(run_dir.rglob("*.sqlite"))
