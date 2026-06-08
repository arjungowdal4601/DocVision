from pathlib import Path

import fitz
import pytest

from vision_indexer.main import main
from tests.fake_llm import fake_build_topic_index_with_llm, fake_process_page_with_llm


def create_pdf(path: Path, page_count: int) -> None:
    document = fitz.open()
    for page_number in range(1, page_count + 1):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {page_number}")
    document.save(path)
    document.close()


def test_pipeline_failure_then_resume_continues_from_failed_page(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    run_dir = tmp_path / "runs" / "failure_resume"
    create_pdf(pdf_path, page_count=2)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("VISION_INDEXER_RETRY_INITIAL_DELAY_SECONDS", "0")
    monkeypatch.setenv("VISION_INDEXER_RETRY_MAX_ATTEMPTS", "2")
    monkeypatch.setattr("vision_indexer.graph.nodes.build_topic_index_with_llm", fake_build_topic_index_with_llm)

    def fail_on_page_two(page_number: int, page_path: Path, framework_memory_md: str, short_term_memory_md: str):
        if page_number == 2:
            raise RuntimeError("Injected fake failure")
        return fake_process_page_with_llm(page_number, page_path, framework_memory_md, short_term_memory_md)

    monkeypatch.setattr("vision_indexer.graph.nodes.process_page_with_llm", fail_on_page_two)

    with pytest.raises(RuntimeError, match="Injected fake failure"):
        main(["--pdf", str(pdf_path), "--out", str(run_dir), "--debug"])

    page_one_output = run_dir / "page_outputs" / "page_0001.json"
    original_page_one = page_one_output.read_text(encoding="utf-8")
    assert page_one_output.exists()
    assert (run_dir / "errors" / "page_0002_error.json").exists()
    assert '"status": "failed"' in (run_dir / "run_status.json").read_text(encoding="utf-8")

    monkeypatch.setattr("vision_indexer.graph.nodes.process_page_with_llm", fake_process_page_with_llm)
    exit_code = main(["--pdf", str(pdf_path), "--out", str(run_dir), "--debug", "--resume"])

    assert exit_code == 0
    assert page_one_output.read_text(encoding="utf-8") == original_page_one
    assert (run_dir / "page_outputs" / "page_0002.json").exists()
    assert (run_dir / "topic_index.json").exists()
    final_status = (run_dir / "run_status.json").read_text(encoding="utf-8")
    assert '"status": "completed"' in final_status
    assert '"completed_pages": [\n    1,\n    2\n  ]' in final_status
