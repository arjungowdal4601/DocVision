from pathlib import Path

import fitz
import pytest

from tests.fake_llm import fake_process_page_with_llm
from vision_indexer.main import main


def create_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Topic index failure fixture")
    document.save(path)
    document.close()


def test_topic_index_failure_marks_run_failed_and_writes_error(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    run_dir = tmp_path / "runs" / "topic_index_failure"
    create_pdf(pdf_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("vision_indexer.graph.nodes.process_page_with_llm", fake_process_page_with_llm)

    def fail_topic_index(*args, **kwargs):
        raise RuntimeError("Injected topic index failure")

    monkeypatch.setattr("vision_indexer.graph.nodes.build_topic_index_with_llm", fail_topic_index)

    with pytest.raises(RuntimeError, match="Injected topic index failure"):
        main(["--pdf", str(pdf_path), "--out", str(run_dir), "--debug"])

    assert (run_dir / "page_outputs" / "page_0001.json").exists()
    assert not (run_dir / "topic_index.json").exists()
    assert (run_dir / "errors" / "topic_index_error.json").exists()
    assert (run_dir / "manifest.json").exists()
    error_text = (run_dir / "errors" / "topic_index_error.json").read_text(encoding="utf-8")
    assert "Injected topic index failure" in error_text
    assert '"batch_number": 1' in error_text
    assert '"page_range": "1-1"' in error_text
    assert '"status": "failed"' in (run_dir / "run_status.json").read_text(encoding="utf-8")
    assert '"status": "failed"' in (run_dir / "manifest.json").read_text(encoding="utf-8")
