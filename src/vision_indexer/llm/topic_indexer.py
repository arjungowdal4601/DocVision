from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_indexer.config import AppConfig
from vision_indexer.llm.provider import build_chat_model
from vision_indexer.prompts.topic_index_prompt import build_topic_index_messages
from vision_indexer.schemas.page_output import PageIndexOutput
from vision_indexer.schemas.topic_index import TopicIndexOutput


def build_topic_index_with_llm(
    page_outputs: list[PageIndexOutput],
    source_pdf_path: Path,
    previous_topic_index: TopicIndexOutput | None = None,
    framework_memory_md: str = "",
    batch_number: int = 1,
    total_batches: int = 1,
    config: AppConfig | None = None,
) -> tuple[TopicIndexOutput, Any]:
    messages = build_topic_index_messages(
        page_outputs=page_outputs,
        source_pdf_path=source_pdf_path,
        previous_topic_index=previous_topic_index,
        framework_memory_md=framework_memory_md,
        batch_number=batch_number,
        total_batches=total_batches,
    )
    llm = build_chat_model(config)
    structured_llm = llm.with_structured_output(TopicIndexOutput, include_raw=True)
    result = structured_llm.invoke(messages)

    parsed = result.get("parsed")
    if parsed is None:
        parsing_error = result.get("parsing_error")
        raise ValueError(f"LLM response did not match TopicIndexOutput: {parsing_error}")

    return parsed, result.get("raw")
