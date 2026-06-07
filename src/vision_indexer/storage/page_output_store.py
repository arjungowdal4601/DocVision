from __future__ import annotations

from pathlib import Path

from vision_indexer.schemas.page_output import PageIndexOutput


class PageOutputStore:
    def __init__(self, run_dir: Path) -> None:
        self.output_dir = run_dir / "page_outputs"

    def save_page_output(self, page_output: PageIndexOutput) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"page_{page_output.page_number:04d}.json"
        output_path.write_text(page_output.model_dump_json(indent=2), encoding="utf-8")
        return output_path
