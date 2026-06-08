from pathlib import Path
import csv

import fitz

from vision_indexer.main import main
from tests.fake_llm import fake_build_topic_index_with_llm, fake_process_page_with_llm


def create_sample_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Pipeline fixture")
    document.save(path)
    document.close()


def test_pipeline_run_writes_expected_run_outputs(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    run_dir = tmp_path / "runs" / "test_run"
    create_sample_pdf(pdf_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("vision_indexer.graph.nodes.process_page_with_llm", fake_process_page_with_llm)
    monkeypatch.setattr("vision_indexer.graph.nodes.build_topic_index_with_llm", fake_build_topic_index_with_llm)

    exit_code = main(["--pdf", str(pdf_path), "--out", str(run_dir), "--debug"])

    assert exit_code == 0
    assert (run_dir / "source" / "source.pdf").exists()
    assert (run_dir / "page_images" / "page_0001.png").exists()
    assert (run_dir / "memories" / "framework_memory.md").exists()
    assert (run_dir / "memories" / "short_term_memory.md").exists()
    assert (run_dir / "memory_debug" / "framework" / "page_0001_before.md").exists()
    assert (run_dir / "page_outputs" / "page_0001.json").exists()
    assert (run_dir / "topic_index.json").exists()
    assert (run_dir / "topic_index_batches" / "batch_0001_topic_index.json").exists()
    assert (run_dir / "graph" / "graph.mmd").exists()
    assert (run_dir / "logs" / "run.log").exists()
    assert (run_dir / "logs" / "tokenomics.log").exists()
    assert (run_dir / "checkpoints" / "page_checkpoints.csv").exists()

    manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
    framework_memory = (run_dir / "memories" / "framework_memory.md").read_text(encoding="utf-8")
    short_term_memory = (run_dir / "memories" / "short_term_memory.md").read_text(encoding="utf-8")
    run_log = (run_dir / "logs" / "run.log").read_text(encoding="utf-8")
    tokenomics_log = (run_dir / "logs" / "tokenomics.log").read_text(encoding="utf-8")
    with (run_dir / "checkpoints" / "page_checkpoints.csv").open(newline="", encoding="utf-8") as handle:
        checkpoint_events = [row["event"] for row in csv.DictReader(handle)]

    assert '"page_count": 1' in manifest_text
    assert '"total_tokens": 40' in manifest_text
    assert '"cached_input_tokens": 8' in manifest_text
    assert '"reasoning_tokens": 3' in manifest_text
    assert '"visible_output_tokens": 7' in manifest_text
    assert "Processed page 1." in framework_memory
    assert "Latest page: 1." in short_term_memory
    assert "Starting topic indexing for 1 page output(s) in 1 batch(es)" in run_log
    assert "Topic index batch 1/1 started for pages 1-1" in run_log
    assert "Saved topic index batch 1" in run_log
    assert "Saved final topic index" in run_log
    assert "operation=process_page_node page_number=1" in tokenomics_log
    assert "operation=build_topic_index_batch page_number=None" in tokenomics_log
    assert "batch_number=1" in tokenomics_log
    assert "page_range=1-1" in tokenomics_log
    assert "provider=openai" in tokenomics_log
    assert "model=gpt-5.4" in tokenomics_log
    assert "reasoning_effort=medium" in tokenomics_log
    assert "cached_input_tokens=3" in tokenomics_log
    assert "reasoning_tokens=1" in tokenomics_log
    assert "visible_output_tokens=3" in tokenomics_log
    assert "operation=run_total" in tokenomics_log
    assert '"topic_name": "Fake Topic"' in (run_dir / "topic_index.json").read_text(encoding="utf-8")
    assert {"run_started", "page_started", "page_completed", "run_completed"}.issubset(checkpoint_events)
