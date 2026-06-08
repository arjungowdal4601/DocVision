from pathlib import Path

from vision_indexer.memory.memory_store import MemoryStore
from vision_indexer.schemas.memory_patch import MarkdownMemoryEdit


def test_memory_store_writes_memories_and_debug_snapshots(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, debug_memory=True)
    store.initialize()

    store.apply_page_memory_update(
        page_number=1,
        framework_memory_edits=[
            MarkdownMemoryEdit(
                edit_type="append_new_section",
                section_heading="## Document Identity",
                content_md="Processed page 1.",
            )
        ],
        short_term_memory_edits=[
            MarkdownMemoryEdit(
                edit_type="append_new_section",
                section_heading="## Active Reading Position",
                content_md="Latest page: 1.",
            )
        ],
    )

    assert "Processed page 1." in store.framework_memory_path.read_text(encoding="utf-8")
    assert "Latest page: 1." in store.short_term_memory_path.read_text(encoding="utf-8")
    assert (tmp_path / "memory_debug" / "framework" / "page_0001_before.md").exists()
    assert (tmp_path / "memory_debug" / "framework" / "page_0001_after.md").exists()
    assert (tmp_path / "memory_debug" / "short_term" / "page_0001_before.md").exists()
    assert (tmp_path / "memory_debug" / "short_term" / "page_0001_after.md").exists()


def test_memory_store_reads_cleaned_framework_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, debug_memory=False)
    store.initialize()
    store.framework_memory_path.write_text(
        """# Framework Memory

## Document Identity
- Existing identity.

## Important Assets
- Figure 3 compares architectures.

## Section Map
- Page 1 has the abstract.
""",
        encoding="utf-8",
    )

    result = store.read_framework_memory()

    assert "## Document Identity" in result
    assert "## Section Map" not in result
    assert "## Important Assets" not in result
    assert "Page 1 has the abstract" not in result
    assert "Figure 3 compares architectures" not in result


def test_memory_store_debug_snapshots_use_cleaned_framework_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, debug_memory=True)
    store.initialize()
    store.framework_memory_path.write_text(
        """# Framework Memory

## Document Identity
- Existing identity.

## Section Map
- Page 1 has the abstract.
""",
        encoding="utf-8",
    )

    store.apply_page_memory_update(
        page_number=2,
        framework_memory_edits=[
            MarkdownMemoryEdit(
                edit_type="append_to_section",
                section_heading="## Recurring Concepts",
                content_md="- Residual learning.",
            )
        ],
        short_term_memory_edits=[MarkdownMemoryEdit(edit_type="no_change", section_heading=None, content_md="")],
    )

    before = (tmp_path / "memory_debug" / "framework" / "page_0002_before.md").read_text(encoding="utf-8")
    after = (tmp_path / "memory_debug" / "framework" / "page_0002_after.md").read_text(encoding="utf-8")

    assert "## Section Map" not in before
    assert "Page 1 has the abstract" not in before
    assert "## Section Map" not in after
    assert "Page 1 has the abstract" not in after
    assert "## Recurring Concepts" in after
