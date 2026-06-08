from __future__ import annotations

from pathlib import Path


def test_no_sqlite_references_in_runtime_code_or_docs() -> None:
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "src",
        root / "README.md",
        root / "pyproject.toml",
        root / "requirements.txt",
    ]
    needles = ("sqlite", "Sqlite", ".sqlite", "graph_state")
    offenders: list[str] = []

    for target in targets:
        paths = target.rglob("*") if target.is_dir() else [target]
        for path in paths:
            if path.is_file() and path.suffix in {".py", ".md", ".toml", ".txt"}:
                text = path.read_text(encoding="utf-8")
                if any(needle in text for needle in needles):
                    offenders.append(str(path.relative_to(root)))

    assert offenders == []
