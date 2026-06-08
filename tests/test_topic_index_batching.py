from pathlib import Path
from types import SimpleNamespace

from vision_indexer.graph.nodes import build_topic_index_node
from vision_indexer.schemas.page_output import Asset, PageIndexOutput, Topic
from vision_indexer.schemas.topic_index import (
    TopicIndexDocument,
    TopicIndexOutput,
    TopicIndexTopic,
)
from vision_indexer.storage.run_status_store import initialize_run_status


def build_page_output(page_number: int) -> PageIndexOutput:
    return PageIndexOutput(
        page_number=page_number,
        page_type="body_content",
        page_image_path=f"page_{page_number:04d}.png",
        index_worthy=True,
        summary=f"Summary for page {page_number}.",
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


def build_topic_index(batch_number: int, total_pages: int) -> TopicIndexOutput:
    return TopicIndexOutput(
        document=TopicIndexDocument(
            title=f"Batch {batch_number}",
            source_pdf="sample.pdf",
            total_pages=total_pages,
        ),
        topics=[
            TopicIndexTopic(
                topic_id=f"T{batch_number:03d}",
                topic_name=f"Batch {batch_number} Topic",
                topic_description=f"Topic index after batch {batch_number}.",
                primary_pages=[],
            )
        ],
        unindexed_pages=[],
    )


def test_build_topic_index_node_batches_pages_and_carries_previous_index(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    page_outputs_dir = run_dir / "page_outputs"
    page_outputs_dir.mkdir(parents=True)
    memory_dir = run_dir / "memories"
    memory_dir.mkdir(parents=True)
    (memory_dir / "framework_memory.md").write_text(
        "# Framework Memory\n\n## Core Claim\nFramework context.",
        encoding="utf-8",
    )
    for page_number in range(1, 26):
        (page_outputs_dir / f"page_{page_number:04d}.json").write_text(
            build_page_output(page_number).model_dump_json(indent=2),
            encoding="utf-8",
        )

    captures = []

    def fake_batch_indexer(
        page_outputs,
        source_pdf_path,
        previous_topic_index=None,
        framework_memory_md="",
        batch_number=1,
        total_batches=1,
        config=None,
    ):
        captures.append(
            {
                "pages": [page.page_number for page in page_outputs],
                "previous_topic_index": previous_topic_index,
                "framework_memory_md": framework_memory_md,
                "batch_number": batch_number,
                "total_batches": total_batches,
            }
        )
        return build_topic_index(batch_number, total_pages=25), SimpleNamespace(
            usage_metadata={"input_tokens": 20, "output_tokens": 6, "total_tokens": 26},
            response_metadata={},
        )

    monkeypatch.setattr("vision_indexer.graph.nodes.build_topic_index_with_llm", fake_batch_indexer)
    run_status = initialize_run_status(
        run_id="run",
        pdf_path="sample.pdf",
        output_dir=str(run_dir),
        page_paths=[f"page_{page_number:04d}.png" for page_number in range(1, 26)],
    )

    result = build_topic_index_node(
        {
            "run_dir": str(run_dir),
            "source_pdf_path": "sample.pdf",
            "run_status": run_status.model_dump(mode="json"),
            "token_usage": {},
            "llm_provider": "openai",
            "model_name": "gpt-5.4",
            "reasoning_effort": "medium",
        }
    )

    assert [capture["pages"] for capture in captures] == [
        list(range(1, 11)),
        list(range(11, 21)),
        list(range(21, 26)),
    ]
    assert captures[0]["previous_topic_index"] is None
    assert captures[1]["previous_topic_index"].document.title == "Batch 1"
    assert captures[2]["previous_topic_index"].document.title == "Batch 2"
    assert all(capture["framework_memory_md"].strip().endswith("Framework context.") for capture in captures)
    assert [capture["batch_number"] for capture in captures] == [1, 2, 3]
    assert all(capture["total_batches"] == 3 for capture in captures)

    batch_dir = run_dir / "topic_index_batches"
    assert (batch_dir / "batch_0001_topic_index.json").exists()
    assert (batch_dir / "batch_0002_topic_index.json").exists()
    assert (batch_dir / "batch_0003_topic_index.json").exists()
    final_topic_index = run_dir / "topic_index.json"
    assert final_topic_index.exists()
    assert final_topic_index.read_text(encoding="utf-8") == (
        batch_dir / "batch_0003_topic_index.json"
    ).read_text(encoding="utf-8")
    assert result["token_usage"]["operations"]["build_topic_index_batch"]["calls"] == 3
    assert result["token_usage"]["total"]["total_tokens"] == 78
