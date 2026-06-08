from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vision_indexer.schemas.llm_response import PageProcessingResponse
from vision_indexer.schemas.memory_patch import MarkdownMemoryEdit, MemoryEditBundle
from vision_indexer.schemas.page_output import Asset, PageIndexOutput, Topic
from vision_indexer.schemas.topic_index import (
    TopicIndexAsset,
    TopicIndexDocument,
    TopicIndexOutput,
    TopicIndexPage,
    TopicIndexTopic,
    UnindexedPage,
)


def build_fake_page_response(page_number: int, page_path: Path) -> PageProcessingResponse:
    return PageProcessingResponse(
        memory_edits=MemoryEditBundle(
            framework_memory_edits=[
                MarkdownMemoryEdit(
                    edit_type="append_to_section",
                    section_heading="## Document Identity",
                    content_md=f"Processed page {page_number}.",
                )
            ],
            short_term_memory_edits=[
                MarkdownMemoryEdit(
                    edit_type="append_to_section",
                    section_heading="## Active Reading Position",
                    content_md=f"Latest page: {page_number}.",
                )
            ],
        ),
        page_index_output=PageIndexOutput(
            page_number=page_number,
            page_image_path=str(page_path),
            page_type="body_content",
            index_worthy=True,
            summary=f"Detailed fake summary for page {page_number}.",
            topics=[
                Topic(
                    topic_id=None,
                    topic_name="Fake Topic",
                    topic_description="Deterministic fake topic for tests.",
                )
            ],
            assets=[
                Asset(
                    asset_id=None,
                    asset_type="image",
                    asset_name=None,
                    asset_description="Rendered page image.",
                )
            ],
        ),
    )


def fake_process_page_with_llm(
    page_number: int,
    page_path: Path,
    framework_memory_md: str,
    short_term_memory_md: str,
):
    return build_fake_page_response(page_number, page_path), SimpleNamespace(
        usage_metadata={
            "input_tokens": 10,
            "input_token_details": {"cache_read": 3},
            "output_tokens": 4,
            "output_token_details": {"reasoning": 1},
            "total_tokens": 14,
        },
        response_metadata={},
    )


def fake_build_topic_index_with_llm(
    page_outputs: list[PageIndexOutput],
    source_pdf_path: Path,
    previous_topic_index: TopicIndexOutput | None = None,
    framework_memory_md: str = "",
    batch_number: int = 1,
    total_batches: int = 1,
    config=None,
):
    total_pages = len(page_outputs)
    first_index_worthy_page = next((page for page in page_outputs if page.index_worthy), None)
    primary_pages = []
    if first_index_worthy_page is not None:
        primary_pages.append(
            TopicIndexPage(
                page_number=first_index_worthy_page.page_number,
                page_image_path=first_index_worthy_page.page_image_path,
                description=f"Routes to fake page {first_index_worthy_page.page_number}.",
                assets=[
                    TopicIndexAsset(
                        asset_id=asset.asset_id,
                        asset_type=asset.asset_type,
                        asset_name=asset.asset_name,
                        asset_description=asset.asset_description,
                    )
                    for asset in first_index_worthy_page.assets
                ],
            )
        )

    return TopicIndexOutput(
        document=TopicIndexDocument(
            title="Fake Document",
            source_pdf=str(source_pdf_path),
            total_pages=total_pages,
            document_description=(
                f"Deterministic fake topic index for batch {batch_number} of {total_batches}. "
                f"Previous index present: {previous_topic_index is not None}. "
                f"Framework memory chars: {len(framework_memory_md)}."
            ),
        ),
        topics=[
            TopicIndexTopic(
                topic_id="T001",
                topic_name="Fake Topic",
                topic_description=(
                    "Merged fake topic for tests covering the first index-worthy page, its rendered "
                    "image path, and any assets copied from that page output."
                ),
                primary_pages=primary_pages,
            )
        ],
        unindexed_pages=[
            UnindexedPage(
                page_number=page.page_number,
                page_type=page.page_type,
                reason="Page was not index-worthy.",
            )
            for page in page_outputs
            if not page.index_worthy
        ],
    ), SimpleNamespace(
        usage_metadata={
            "input_tokens": 20,
            "input_token_details": {"cache_read": 5},
            "output_tokens": 6,
            "output_token_details": {"reasoning": 2},
            "total_tokens": 26,
        },
        response_metadata={},
    )
