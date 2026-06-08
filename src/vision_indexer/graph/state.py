from __future__ import annotations

from typing import Any, TypedDict


class VisionIndexerState(TypedDict, total=False):
    source_pdf_path: str
    run_source_pdf_path: str
    run_dir: str
    dpi: int
    debug_memory: bool
    resume: bool
    force: bool
    force_render: bool
    force_pages: list[int]
    max_pages: int | None
    retry_policy: dict[str, Any]
    llm_provider: str
    model_name: str
    reasoning_effort: str
    page_image_paths: list[str]
    current_page_index: int
    framework_memory_md: str
    short_term_memory_md: str
    current_page_output: dict[str, Any]
    pending_framework_memory_edits: list[dict[str, Any]]
    pending_short_term_memory_edits: list[dict[str, Any]]
    page_action: str
    run_status: dict[str, Any]
    page_statuses: dict[str, Any]
    failed_pages: list[int]
    page_results: list[str]
    completed_pages: int
    completed_page_numbers: list[int]
    token_usage: dict[str, Any]
    topic_index_path: str
    next_step: str
    manifest: dict[str, Any]
