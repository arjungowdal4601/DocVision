from pathlib import Path

from vision_indexer.prompts.topic_index_prompt import SYSTEM_PROMPT, build_topic_index_messages
from vision_indexer.schemas.page_output import PageIndexOutput, Topic
from vision_indexer.schemas.topic_index import TopicIndexDocument, TopicIndexOutput, TopicIndexTopic


def test_topic_index_prompt_defines_routing_index_contract() -> None:
    prompt = SYSTEM_PROMPT

    assert "topic_index.json" in prompt
    assert "final navigation map" in prompt
    assert "merge duplicate or similar page-level topics" in prompt
    assert "primary_pages" in prompt
    assert "page_image_path" in prompt
    assert "description" in prompt
    assert "assets" in prompt
    assert "asset_description" in prompt
    assert "Do not use aliases" in prompt
    assert "create aliases" not in prompt
    assert "factual only" in prompt
    assert "no vague wording" in prompt
    assert "methods, datasets, equations, results, figures, or tables" in prompt
    assert "unindexed_pages" in prompt
    assert "key_assets" in prompt
    assert "why_this_page" in prompt
    assert "supporting_pages" in prompt
    assert "Do not use key_assets" in prompt
    assert "Do not use why_this_page" in prompt
    assert "Do not use supporting_pages" in prompt
    assert "previous topic index" in prompt
    assert "latest framework memory" in prompt
    assert "current batch page outputs" in prompt
    assert "Do not resend or require old page JSON files" in prompt


def test_topic_index_messages_include_previous_index_framework_memory_and_current_batch_only() -> None:
    previous_topic_index = TopicIndexOutput(
        document=TopicIndexDocument(title="Previous Map", source_pdf="paper.pdf", total_pages=10),
        topics=[
            TopicIndexTopic(
                topic_id="T001",
                topic_name="Existing Topic",
                topic_description="Existing topic description.",
                primary_pages=[],
            )
        ],
        unindexed_pages=[],
    )
    page_output = PageIndexOutput(
        page_number=11,
        page_type="body_content",
        page_image_path="page_0011.png",
        index_worthy=True,
        summary="Current batch page.",
        topics=[
            Topic(
                topic_id=None,
                topic_name="New Topic",
                topic_description="New topic description.",
            )
        ],
        assets=[],
    )

    messages = build_topic_index_messages(
        page_outputs=[page_output],
        source_pdf_path=Path("paper.pdf"),
        previous_topic_index=previous_topic_index,
        framework_memory_md="# Framework Memory\n\n## Core Claim\nExisting context.",
        batch_number=2,
        total_batches=3,
    )

    human_text = messages[1].content
    assert "Batch number: 2 of 3" in human_text
    assert "Previous topic index JSON" in human_text
    assert "Existing Topic" in human_text
    assert "Latest framework memory Markdown" in human_text
    assert "Existing context" in human_text
    assert "Current batch page outputs JSON" in human_text
    assert "New Topic" in human_text
    assert "page_0010" not in human_text
