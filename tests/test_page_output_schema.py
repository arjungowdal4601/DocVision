import pytest
from pydantic import ValidationError

from vision_indexer.schemas.page_output import Asset, PageIndexOutput, Topic


def test_page_output_schema_accepts_simplified_page_level_shape() -> None:
    output = PageIndexOutput(
        page_number=1,
        page_image_path="runs/test/page_images/page_0001.png",
        page_type="mock",
        index_worthy=True,
        summary="Detailed mock summary describing what is on this page.",
        topics=[
            Topic(
                topic_id=None,
                topic_name="Mock Topic",
                topic_description="Mock topic description.",
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
    )

    dumped = output.model_dump()

    assert dumped["page_image_path"] == "runs/test/page_images/page_0001.png"
    assert dumped["summary"] == "Detailed mock summary describing what is on this page."
    assert dumped["topics"][0]["topic_id"] is None
    assert dumped["topics"][0]["topic_name"] == "Mock Topic"
    assert dumped["topics"][0]["topic_description"] == "Mock topic description."
    assert dumped["assets"][0]["asset_name"] is None
    assert dumped["assets"][0]["asset_description"] == "Rendered page image."


def test_page_output_schema_rejects_removed_old_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        PageIndexOutput.model_validate(
            {
                "page_number": 1,
                "page_path": "runs/test/page_images/page_0001.png",
                "page_type": "mock",
                "index_worthy": True,
                "sections": [],
                "titles": [],
                "topics": [
                    {
                        "topic_id": None,
                        "source_section_id": None,
                        "topic": "Old topic",
                        "description": "Old topic description.",
                    }
                ],
                "assets": [
                    {
                        "asset_id": None,
                        "asset_type": "image",
                        "asset_title": None,
                        "description": "Old asset description.",
                    }
                ],
                "brief_summary": "Old summary.",
            }
        )

    error_text = str(exc_info.value)
    assert "page_image_path" in error_text
    assert "summary" in error_text
    assert "sections" in error_text
    assert "titles" in error_text
    assert "brief_summary" in error_text
