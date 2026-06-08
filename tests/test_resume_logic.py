from pathlib import Path
import csv

import fitz

from vision_indexer.main import main
from vision_indexer.storage.run_status_store import (
    initialize_run_status,
    save_run_status,
    update_page_status,
)
from tests.fake_llm import fake_build_topic_index_with_llm, fake_process_page_with_llm
from vision_indexer.schemas.page_output import PageIndexOutput


def create_pdf(path: Path, page_count: int) -> None:
    document = fitz.open()
    for page_number in range(1, page_count + 1):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {page_number}")
    document.save(path)
    document.close()


def test_resume_skips_completed_page_and_processes_next_page(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    run_dir = tmp_path / "runs" / "resume_run"
    create_pdf(pdf_path, page_count=2)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("vision_indexer.graph.nodes.process_page_with_llm", fake_process_page_with_llm)
    monkeypatch.setattr("vision_indexer.graph.nodes.build_topic_index_with_llm", fake_build_topic_index_with_llm)

    page_output_dir = run_dir / "page_outputs"
    page_output_dir.mkdir(parents=True)
    page_one_output = page_output_dir / "page_0001.json"
    page_one_output.write_text(
        PageIndexOutput(
            page_number=1,
            page_type="body_content",
            page_image_path="page_0001.png",
            index_worthy=True,
            summary="Existing sentinel summary, do not overwrite.",
            topics=[],
            assets=[],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    memory_dir = run_dir / "memories"
    memory_dir.mkdir(parents=True)
    (memory_dir / "framework_memory.md").write_text("# Framework Memory\n\nExisting.\n", encoding="utf-8")
    (memory_dir / "short_term_memory.md").write_text("# Short-Term Memory\n\nExisting.\n", encoding="utf-8")

    status = initialize_run_status(
        run_id="resume_run",
        pdf_path=str(pdf_path),
        output_dir=str(run_dir),
        page_paths=["page_0001.png", "page_0002.png"],
    )
    status = update_page_status(
        status,
        page_number=1,
        status="completed",
        output_path=str(page_one_output),
    )
    save_run_status(run_dir, status)

    exit_code = main(["--pdf", str(pdf_path), "--out", str(run_dir), "--debug", "--resume"])

    assert exit_code == 0
    assert "Existing sentinel summary, do not overwrite." in page_one_output.read_text(encoding="utf-8")
    assert (page_output_dir / "page_0002.json").exists()
    assert (run_dir / "topic_index.json").exists()
    run_status = (run_dir / "run_status.json").read_text(encoding="utf-8")
    with (run_dir / "checkpoints" / "page_checkpoints.csv").open(newline="", encoding="utf-8") as handle:
        checkpoint_events = [row["event"] for row in csv.DictReader(handle)]

    assert '"status": "skipped"' in run_status
    assert '"completed_pages": [\n    1,\n    2\n  ]' in run_status
    assert "run_resumed" in checkpoint_events
    assert "page_skipped" in checkpoint_events
    assert checkpoint_events.count("page_completed") == 1
