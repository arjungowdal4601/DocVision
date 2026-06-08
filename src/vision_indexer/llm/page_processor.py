from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_indexer.config import AppConfig
from vision_indexer.llm.image_input import build_image_input
from vision_indexer.llm.provider import build_chat_model
from vision_indexer.prompts.page_processing_prompt import build_page_processing_messages
from vision_indexer.schemas.llm_response import PageProcessingResponse


def process_page_with_llm(
    page_number: int,
    page_path: Path,
    framework_memory_md: str,
    short_term_memory_md: str,
    config: AppConfig | None = None,
) -> tuple[PageProcessingResponse, Any]:
    image_input = build_image_input(page_path)
    messages = build_page_processing_messages(
        page_number=page_number,
        page_path=str(page_path),
        framework_memory_md=framework_memory_md,
        short_term_memory_md=short_term_memory_md,
        image_input=image_input,
    )
    llm = build_chat_model(config)
    structured_llm = llm.with_structured_output(PageProcessingResponse, include_raw=True)
    result = structured_llm.invoke(messages)

    parsed = result.get("parsed")
    if parsed is None:
        parsing_error = result.get("parsing_error")
        raise ValueError(f"LLM response did not match PageProcessingResponse: {parsing_error}")

    return parsed, result.get("raw")
