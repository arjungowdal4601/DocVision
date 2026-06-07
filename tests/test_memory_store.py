from pathlib import Path

from vision_indexer.memory.memory_store import MemoryStore


def test_memory_store_writes_memories_and_debug_snapshots(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, debug_memory=True)
    store.initialize()

    store.apply_page_memory_update(
        page_number=1,
        framework_update_md="# Framework Memory\n\nProcessed page 1.\n",
        short_term_update_md="# Short-Term Memory\n\nLatest page: 1.\n",
    )

    assert store.framework_memory_path.read_text(encoding="utf-8").startswith("# Framework Memory")
    assert store.short_term_memory_path.read_text(encoding="utf-8").startswith("# Short-Term Memory")
    assert (tmp_path / "memory_debug" / "framework" / "page_0001_before.md").exists()
    assert (tmp_path / "memory_debug" / "framework" / "page_0001_after.md").exists()
    assert (tmp_path / "memory_debug" / "short_term" / "page_0001_before.md").exists()
    assert (tmp_path / "memory_debug" / "short_term" / "page_0001_after.md").exists()
