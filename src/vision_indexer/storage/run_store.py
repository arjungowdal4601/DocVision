from __future__ import annotations

import shutil
from pathlib import Path


class RunStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def prepare_directories(self) -> None:
        for relative_dir in (
            "source",
            "page_images",
            "memories",
            "memory_debug/framework",
            "memory_debug/short_term",
            "page_outputs",
            "graph",
            "logs",
        ):
            (self.run_dir / relative_dir).mkdir(parents=True, exist_ok=True)

    def copy_source_pdf(self, source_pdf_path: Path) -> Path:
        self.prepare_directories()
        destination = self.run_dir / "source" / "source.pdf"
        if source_pdf_path.resolve() != destination.resolve():
            shutil.copy2(source_pdf_path, destination)
        return destination
