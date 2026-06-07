from vision_indexer.schemas.page_output import Asset, PageIndexOutput, Section, Title, Topic


def test_page_output_schema_accepts_page_level_topic_without_topic_id() -> None:
    output = PageIndexOutput(
        page_number=1,
        page_path="runs/test/page_images/page_0001.png",
        page_type="mock",
        index_worthy=True,
        sections=[
            Section(
                section_id="section-0001-001",
                source_section_id=None,
                heading="Mock Section",
                text="Mock section text.",
            )
        ],
        titles=[Title(title_id="title-0001-001", text="Mock Title", level=1)],
        topics=[
            Topic(
                topic_id=None,
                source_section_id="section-0001-001",
                label="Mock Topic",
                confidence=1.0,
            )
        ],
        assets=[
            Asset(
                asset_id="asset-0001-001",
                source_section_id="section-0001-001",
                asset_type="page_image",
                description="Rendered page image.",
            )
        ],
        brief_summary="Mock summary.",
    )

    dumped = output.model_dump()

    assert dumped["topics"][0]["topic_id"] is None
    assert dumped["sections"][0]["source_section_id"] is None
