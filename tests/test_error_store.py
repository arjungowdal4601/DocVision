from pathlib import Path

from vision_indexer.storage.error_store import write_page_error


def test_write_page_error_creates_json_record(tmp_path: Path) -> None:
    error_path = write_page_error(tmp_path, page_number=53, error=ValueError("bad page"), attempt=3)

    content = error_path.read_text(encoding="utf-8")

    assert error_path == tmp_path / "errors" / "page_0053_error.json"
    assert '"page_number": 53' in content
    assert '"error_type": "ValueError"' in content
    assert '"error_message": "bad page"' in content
    assert '"attempt": 3' in content
