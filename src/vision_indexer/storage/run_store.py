from __future__ import annotations

import shutil
from pathlib import Path


RUN_SUBDIRS = (
    "source",
    "page_images",
    "memories",
    "memory_debug/framework",
    "memory_debug/short_term",
    "page_outputs",
    "topic_index_batches",
    "graph",
    "logs",
    "errors",
    "checkpoints",
)
RUN_FILES = ("manifest.json", "run_status.json", "topic_index.json")


class RunStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def prepare_directories(self) -> None:
        for relative_dir in RUN_SUBDIRS:
            (self.run_dir / relative_dir).mkdir(parents=True, exist_ok=True)

    def reset_run_outputs(self) -> None:
        for relative_dir in RUN_SUBDIRS:
            target = self.run_dir / relative_dir
            if target.exists():
                shutil.rmtree(target)
        for filename in RUN_FILES:
            target_file = self.run_dir / filename
            if target_file.exists():
                target_file.unlink()

    def copy_source_pdf(self, source_pdf_path: Path) -> Path:
        self.prepare_directories()
        destination = self.run_dir / "source" / "source.pdf"
        if source_pdf_path.resolve() != destination.resolve():
            shutil.copy2(source_pdf_path, destination)
        return destination
