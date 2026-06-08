from __future__ import annotations

from pathlib import Path

from vision_indexer.schemas.page_output import PageIndexOutput
from vision_indexer.schemas.topic_index import TopicIndexOutput


class TopicIndexStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.topic_index_path = run_dir / "topic_index.json"
        self.topic_index_batches_dir = run_dir / "topic_index_batches"
        self.page_outputs_dir = run_dir / "page_outputs"

    def save_topic_index(self, topic_index: TopicIndexOutput) -> Path:
        self.topic_index_path.write_text(topic_index.model_dump_json(indent=2), encoding="utf-8")
        return self.topic_index_path

    def save_batch_topic_index(self, batch_number: int, topic_index: TopicIndexOutput) -> Path:
        self.topic_index_batches_dir.mkdir(parents=True, exist_ok=True)
        batch_path = self.topic_index_batches_dir / f"batch_{batch_number:04d}_topic_index.json"
        batch_path.write_text(topic_index.model_dump_json(indent=2), encoding="utf-8")
        return batch_path

    def load_page_outputs(self) -> list[PageIndexOutput]:
        page_outputs = [
            PageIndexOutput.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.page_outputs_dir.glob("page_*.json"))
        ]
        return sorted(page_outputs, key=lambda page_output: page_output.page_number)
