from vision_indexer.schemas.llm_response import PageProcessingResponse


def test_page_processing_response_uses_memory_edits_not_full_memory() -> None:
    response = PageProcessingResponse.model_validate(
        {
            "memory_edits": {
                "framework_memory_edits": [
                    {
                        "edit_type": "append_new_section",
                        "section_heading": "## Document Identity",
                        "content_md": "Research paper about residual learning.",
                    }
                ],
                "short_term_memory_edits": [
                    {
                        "edit_type": "replace_section",
                        "section_heading": "## Active Reading Position",
                        "content_md": "Reading page 1.",
                    }
                ],
            },
            "page_index_output": {
                "page_number": 1,
                "page_image_path": "runs/test/page_images/page_0001.png",
                "page_type": "research_content",
                "index_worthy": True,
                "summary": "Title and abstract page with enough detail for page-level extraction.",
                "topics": [
                    {
                        "topic_id": None,
                        "topic_name": "Title and abstract",
                        "topic_description": "Identifies the paper and summarizes its opening claim.",
                    }
                ],
                "assets": [],
            },
        }
    )

    dumped = response.model_dump()

    assert "framework_memory_md" not in dumped
    assert dumped["memory_edits"]["framework_memory_edits"][0]["section_heading"] == "## Document Identity"
