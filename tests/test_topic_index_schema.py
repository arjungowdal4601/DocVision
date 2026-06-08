import pytest
from pydantic import ValidationError

from vision_indexer.schemas.topic_index import (
    TopicIndexAsset,
    TopicIndexDocument,
    TopicIndexOutput,
    TopicIndexPage,
    TopicIndexTopic,
    UnindexedPage,
)


def test_topic_index_schema_accepts_nested_page_assets() -> None:
    output = TopicIndexOutput(
        document=TopicIndexDocument(
            title="Deep Residual Learning for Image Recognition",
            source_pdf="1512.03385v1.pdf",
            total_pages=12,
            document_description="Paper about residual networks.",
        ),
        topics=[
            TopicIndexTopic(
                topic_id="T001",
                topic_name="Residual learning",
                topic_description=(
                    "Residual learning reformulates a layer stack as a residual function F(x) added "
                    "to the identity input x, which lets very deep convolutional networks optimize "
                    "more reliably than plain stacked layers."
                ),
                primary_pages=[
                    TopicIndexPage(
                        page_number=3,
                        page_image_path="runs/test/page_images/page_0003.png",
                        description="Defines residual learning.",
                        assets=[
                            TopicIndexAsset(
                                asset_id="Equation 1",
                                asset_type="equation",
                                asset_name="Residual block equation",
                                asset_description="Defines the residual block formula.",
                            )
                        ],
                    )
                ],
            )
        ],
        unindexed_pages=[UnindexedPage(page_number=9, page_type="references", reason="References only.")],
    )

    dumped = output.model_dump()

    assert dumped["topics"][0]["primary_pages"][0]["description"] == "Defines residual learning."
    assert dumped["topics"][0]["primary_pages"][0]["assets"][0]["asset_id"] == "Equation 1"
    assert "aliases" not in dumped["topics"][0]
    assert "key_assets" not in dumped["topics"][0]
    assert "supporting_pages" not in dumped["topics"][0]
    assert "why_this_page" not in dumped["topics"][0]["primary_pages"][0]


def test_topic_index_schema_rejects_banned_topic_level_and_page_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TopicIndexOutput.model_validate(
            {
                "document": {
                    "title": "Paper",
                    "source_pdf": "paper.pdf",
                    "total_pages": 1,
                },
                "topics": [
                    {
                        "topic_id": "T001",
                        "topic_name": "Residual learning",
                        "topic_description": "Core idea.",
                        "aliases": ["identity mapping"],
                        "primary_pages": [
                            {
                                "page_number": 1,
                                "page_image_path": "page_0001.png",
                                "why_this_page": "Old field.",
                                "assets": [],
                            }
                        ],
                        "key_assets": [],
                        "supporting_pages": [],
                    }
                ],
                "unindexed_pages": [],
            }
        )

    error_text = str(exc_info.value)
    assert "aliases" in error_text
    assert "why_this_page" in error_text
    assert "key_assets" in error_text
    assert "supporting_pages" in error_text
