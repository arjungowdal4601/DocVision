from __future__ import annotations

import csv
from pathlib import Path

import fitz
import pytest

from tests.fake_llm import fake_process_page_with_llm
from vision_indexer.main import main


def create_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Save failure fixture")
    document.save(path)
    document.close()


def test_save_page_failure_marks_run_failed_and_writes_error(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    run_dir = tmp_path / "runs" / "save_failure"
    create_pdf(pdf_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("vision_indexer.graph.nodes.process_page_with_llm", fake_process_page_with_llm)

    def fail_memory_update(*args, **kwargs) -> None:
        raise RuntimeError("Injected save failure")

    monkeypatch.setattr("vision_indexer.memory.memory_store.MemoryStore.apply_page_memory_update", fail_memory_update)

    with pytest.raises(RuntimeError, match="Injected save failure"):
        main(["--pdf", str(pdf_path), "--out", str(run_dir), "--debug"])

    assert (run_dir / "errors" / "page_0001_error.json").exists()
    assert (run_dir / "manifest.json").exists()
    run_status = (run_dir / "run_status.json").read_text(encoding="utf-8")
    manifest = (run_dir / "manifest.json").read_text(encoding="utf-8")
    with (run_dir / "checkpoints" / "page_checkpoints.csv").open(newline="", encoding="utf-8") as handle:
        checkpoint_events = [row["event"] for row in csv.DictReader(handle)]

    assert '"status": "failed"' in run_status
    assert '"status": "failed"' in manifest
    assert "Injected save failure" in (run_dir / "errors" / "page_0001_error.json").read_text(encoding="utf-8")
    assert "page_failed" in checkpoint_events
