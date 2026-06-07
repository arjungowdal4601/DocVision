from __future__ import annotations

from typing import Any, TypedDict


class VisionIndexerState(TypedDict, total=False):
    source_pdf_path: str
    run_source_pdf_path: str
    run_dir: str
    dpi: int
    debug_memory: bool
    page_image_paths: list[str]
    current_page_index: int
    current_response: dict[str, Any]
    page_results: list[str]
    token_usage: dict[str, Any]
    next_step: str
    manifest: dict[str, Any]
