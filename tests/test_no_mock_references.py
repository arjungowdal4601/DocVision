from __future__ import annotations

from pathlib import Path


def test_no_public_mock_llm_references_remain() -> None:
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "src",
        root / "main.py",
        root / "README.md",
    ]
    needles = ("mock-llm", "mock_llm", "mock_page_processor", "process_page_with_mock")
    offenders: list[str] = []

    for target in targets:
        paths = target.rglob("*") if target.is_dir() else [target]
        for path in paths:
            if path.is_file() and path.suffix in {".py", ".md"}:
                text = path.read_text(encoding="utf-8")
                if any(needle in text for needle in needles):
                    offenders.append(str(path.relative_to(root)))

    assert offenders == []
