from __future__ import annotations

from pathlib import Path


def test_pyproject_is_kept_and_runtime_unused_gitattributes_is_removed() -> None:
    project_root = Path(__file__).resolve().parents[1]

    assert (project_root / "pyproject.toml").exists()
    assert not (project_root / ".gitattributes").exists()
