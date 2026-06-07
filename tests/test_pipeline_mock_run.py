from pathlib import Path

import fitz

from vision_indexer.main import main


def create_sample_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Pipeline fixture")
    document.save(path)
    document.close()


def test_pipeline_mock_run_writes_expected_run_outputs(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    run_dir = tmp_path / "runs" / "test_run"
    create_sample_pdf(pdf_path)

    exit_code = main(["--pdf", str(pdf_path), "--out", str(run_dir), "--debug"])

    assert exit_code == 0
    assert (run_dir / "source" / "source.pdf").exists()
    assert (run_dir / "page_images" / "page_0001.png").exists()
    assert (run_dir / "memories" / "framework_memory.md").exists()
    assert (run_dir / "memories" / "short_term_memory.md").exists()
    assert (run_dir / "memory_debug" / "framework" / "page_0001_before.md").exists()
    assert (run_dir / "page_outputs" / "page_0001.json").exists()
    assert (run_dir / "graph" / "graph.mmd").exists()
    assert (run_dir / "logs" / "run.log").exists()
    assert (run_dir / "logs" / "tokenomics.log").exists()

    manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert '"page_count": 1' in manifest_text
    assert '"total_tokens": 15' in manifest_text
