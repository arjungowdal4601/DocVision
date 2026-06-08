from __future__ import annotations

from pathlib import Path

from vision_indexer.memory.memory_editor import (
    apply_framework_memory_edits,
    apply_short_term_memory_edits,
)
from vision_indexer.schemas.memory_patch import MarkdownMemoryEdit


FRAMEWORK_MEMORY_INITIAL = "# Framework Memory\n\n"
SHORT_TERM_MEMORY_INITIAL = "# Short-Term Memory\n\n"


class MemoryStore:
    def __init__(self, run_dir: Path, debug_memory: bool = False) -> None:
        self.run_dir = run_dir
        self.debug_memory = debug_memory
        self.memories_dir = run_dir / "memories"
        self.framework_memory_path = self.memories_dir / "framework_memory.md"
        self.short_term_memory_path = self.memories_dir / "short_term_memory.md"
        self.framework_debug_dir = run_dir / "memory_debug" / "framework"
        self.short_term_debug_dir = run_dir / "memory_debug" / "short_term"

    def initialize(self) -> None:
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.framework_debug_dir.mkdir(parents=True, exist_ok=True)
        self.short_term_debug_dir.mkdir(parents=True, exist_ok=True)
        self.framework_memory_path.write_text(FRAMEWORK_MEMORY_INITIAL, encoding="utf-8")
        self.short_term_memory_path.write_text(SHORT_TERM_MEMORY_INITIAL, encoding="utf-8")

    def read_framework_memory(self) -> str:
        return apply_framework_memory_edits(
            self.framework_memory_path.read_text(encoding="utf-8"),
            [],
        )

    def read_short_term_memory(self) -> str:
        return self.short_term_memory_path.read_text(encoding="utf-8")

    def apply_page_memory_update(
        self,
        page_number: int,
        framework_memory_edits: list[MarkdownMemoryEdit],
        short_term_memory_edits: list[MarkdownMemoryEdit],
    ) -> None:
        existing_framework = self.read_framework_memory()
        existing_short_term = self.read_short_term_memory()

        if self.debug_memory:
            self._write_debug_snapshot("framework", page_number, "before", existing_framework)
            self._write_debug_snapshot("short_term", page_number, "before", existing_short_term)

        updated_framework = apply_framework_memory_edits(existing_framework, framework_memory_edits)
        updated_short_term = apply_short_term_memory_edits(existing_short_term, short_term_memory_edits)

        self.framework_memory_path.write_text(updated_framework, encoding="utf-8")
        self.short_term_memory_path.write_text(updated_short_term, encoding="utf-8")

        if self.debug_memory:
            self._write_debug_snapshot("framework", page_number, "after", updated_framework)
            self._write_debug_snapshot("short_term", page_number, "after", updated_short_term)

    def _write_debug_snapshot(
        self,
        memory_name: str,
        page_number: int,
        timing: str,
        content: str,
    ) -> None:
        target_dir = self.framework_debug_dir if memory_name == "framework" else self.short_term_debug_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"page_{page_number:04d}_{timing}.md"
        target_path.write_text(content, encoding="utf-8")
