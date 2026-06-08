from pathlib import Path

from vision_indexer.schemas.page_output import Asset, PageIndexOutput, Topic
from vision_indexer.schemas.topic_index import (
    TopicIndexDocument,
    TopicIndexOutput,
    TopicIndexTopic,
)
from vision_indexer.storage.topic_index_store import TopicIndexStore


def build_page_output(page_number: int) -> PageIndexOutput:
    return PageIndexOutput(
        page_number=page_number,
        page_type="body_content",
        page_image_path=f"page_{page_number:04d}.png",
        index_worthy=True,
        summary=f"Summary {page_number}.",
        topics=[
            Topic(
                topic_id=None,
                topic_name=f"Topic {page_number}",
                topic_description=f"Topic {page_number} description.",
            )
        ],
        assets=[
            Asset(
                asset_id=None,
                asset_type="image",
                asset_name=None,
                asset_description="Rendered image.",
            )
        ],
    )


def test_topic_index_store_saves_topic_index_json(tmp_path: Path) -> None:
    store = TopicIndexStore(tmp_path)
    topic_index = TopicIndexOutput(
        document=TopicIndexDocument(title="Paper", source_pdf="paper.pdf", total_pages=1),
        topics=[
            TopicIndexTopic(
                topic_id="T001",
                topic_name="Topic 1",
                topic_description="Topic description.",
                primary_pages=[],
            )
        ],
        unindexed_pages=[],
    )

    output_path = store.save_topic_index(topic_index)

    assert output_path == tmp_path / "topic_index.json"
    assert '"topic_name": "Topic 1"' in output_path.read_text(encoding="utf-8")


def test_topic_index_store_saves_batch_topic_index_json(tmp_path: Path) -> None:
    store = TopicIndexStore(tmp_path)
    topic_index = TopicIndexOutput(
        document=TopicIndexDocument(title="Paper", source_pdf="paper.pdf", total_pages=1),
        topics=[
            TopicIndexTopic(
                topic_id="T001",
                topic_name="Topic 1",
                topic_description="Topic description.",
                primary_pages=[],
            )
        ],
        unindexed_pages=[],
    )

    output_path = store.save_batch_topic_index(batch_number=2, topic_index=topic_index)

    assert output_path == tmp_path / "topic_index_batches" / "batch_0002_topic_index.json"
    assert '"topic_name": "Topic 1"' in output_path.read_text(encoding="utf-8")


def test_topic_index_store_loads_page_outputs_sorted_by_page_number(tmp_path: Path) -> None:
    page_output_dir = tmp_path / "page_outputs"
    page_output_dir.mkdir(parents=True)
    (page_output_dir / "page_0002.json").write_text(
        build_page_output(2).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (page_output_dir / "page_0001.json").write_text(
        build_page_output(1).model_dump_json(indent=2),
        encoding="utf-8",
    )

    page_outputs = TopicIndexStore(tmp_path).load_page_outputs()

    assert [output.page_number for output in page_outputs] == [1, 2]
