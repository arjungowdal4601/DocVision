from __future__ import annotations

from pathlib import Path

from vision_indexer.prompts import page_processing_prompt
from vision_indexer.prompts.page_processing_prompt import SYSTEM_PROMPT


def test_page_processing_prompt_contains_complete_output_contract() -> None:
    prompt = SYSTEM_PROMPT

    assert "FRAMEWORK MEMORY" in prompt
    assert "SHORT-TERM MEMORY" in prompt
    assert "PAGE INDEX OUTPUT" in prompt
    assert "memory_edits" in prompt
    assert "page_index_output" in prompt
    assert "section_heading must be a Markdown heading" in prompt
    assert "## Document Identity" in prompt
    assert "page_image_path" in prompt
    assert "summary" in prompt
    assert "topic_name" in prompt
    assert "topic_description" in prompt
    assert "asset_name" in prompt
    assert "asset_description" in prompt
    assert "topic_id" in prompt
    assert "Do not use old field names" in prompt
    assert "sections" in prompt
    assert "titles" in prompt
    assert "section_path" in prompt
    assert "source_section_id" in prompt
    assert "brief_summary" in prompt
    assert "- page_path" not in prompt
    assert "- sections" not in prompt
    assert "- titles" not in prompt
    assert "- brief_summary" not in prompt
    assert "sections is an array" not in prompt
    assert "titles is an array" not in prompt


def test_framework_memory_prompt_limits_processing_context_sections() -> None:
    prompt = page_processing_prompt.FRAMEWORK_MEMORY_PROMPT

    assert "only for page extraction context" in prompt
    assert "not the final navigation index" in prompt
    assert "Only these framework memory sections are allowed" in prompt
    assert "## Document Identity" in prompt
    assert "## Core Claim" in prompt
    assert "## Recurring Concepts" in prompt
    assert "- ## Section Map" not in prompt
    assert "Do not create ## Section Map" in prompt
    assert "Do not store page-by-page summaries" in prompt
    assert "Do not store navigation maps" in prompt
    assert "Do not create ## Formulas Used Across Document" in prompt
    assert "Do not create ## Important Assets" in prompt
    assert "Formulas, tables, figures, and assets belong in page_index_output" in prompt
    assert "- formulas used across the document" not in prompt
    assert "- important tables, figures, or assets" not in prompt


def test_page_processing_prompt_is_split_into_named_sections() -> None:
    sections = [
        page_processing_prompt.PROMPT_HEADER,
        page_processing_prompt.FRAMEWORK_MEMORY_PROMPT,
        page_processing_prompt.SHORT_TERM_MEMORY_PROMPT,
        page_processing_prompt.PAGE_INDEX_OUTPUT_PROMPT,
        page_processing_prompt.STRICT_RESPONSE_RULES_PROMPT,
    ]

    cursor = 0
    for section in sections:
        assert section.strip()
        index = SYSTEM_PROMPT.index(section, cursor)
        assert index >= cursor
        cursor = index + len(section)


def test_unused_placeholder_prompt_files_are_removed() -> None:
    prompts_dir = Path(page_processing_prompt.__file__).parent

    assert not (prompts_dir / "framework_memory_prompt.py").exists()
    assert not (prompts_dir / "short_term_memory_prompt.py").exists()
    assert not (prompts_dir / "page_indexing_prompt.py").exists()
