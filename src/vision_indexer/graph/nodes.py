from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from vision_indexer.graph.state import VisionIndexerState
from vision_indexer.memory.memory_store import MemoryStore
from vision_indexer.schemas.llm_response import PageProcessingResponse
from vision_indexer.schemas.page_output import Asset, PageIndexOutput, Section, Title, Topic
from vision_indexer.storage.manifest_store import ManifestStore
from vision_indexer.storage.page_output_store import PageOutputStore
from vision_indexer.storage.run_store import RunStore
from vision_indexer.tokenomics.token_tracker import TokenTracker
from vision_indexer.utils.pdf_renderer import render_pdf_to_images


logger = logging.getLogger(__name__)


def initialize_run_node(state: VisionIndexerState) -> VisionIndexerState:
    run_dir = Path(state["run_dir"])
    source_pdf_path = Path(state["source_pdf_path"])
    debug_memory = bool(state.get("debug_memory", False))

    run_store = RunStore(run_dir)
    run_store.prepare_directories()
    copied_pdf_path = run_store.copy_source_pdf(source_pdf_path)
    MemoryStore(run_dir, debug_memory=debug_memory).initialize()

    logger.info("Initialized run at %s", run_dir)
    return {
        "run_source_pdf_path": str(copied_pdf_path),
        "page_image_paths": [],
        "current_page_index": 0,
        "page_results": [],
        "token_usage": TokenTracker.empty_state(),
    }


def render_pdf_node(state: VisionIndexerState) -> VisionIndexerState:
    run_dir = Path(state["run_dir"])
    pdf_path = Path(state["run_source_pdf_path"])
    dpi = int(state.get("dpi", 150))

    page_image_paths = render_pdf_to_images(pdf_path, run_dir / "page_images", dpi=dpi)
    if not page_image_paths:
        raise ValueError(f"PDF contains no pages: {pdf_path}")

    logger.info("Rendered %s PDF page(s)", len(page_image_paths))
    return {
        "page_image_paths": [str(path) for path in page_image_paths],
        "current_page_index": 0,
    }


def process_page_node(state: VisionIndexerState) -> VisionIndexerState:
    page_image_paths = state["page_image_paths"]
    current_page_index = int(state.get("current_page_index", 0))
    if current_page_index >= len(page_image_paths):
        raise IndexError("Current page index is outside rendered page range")

    page_number = current_page_index + 1
    page_path = page_image_paths[current_page_index]
    response = _build_mock_page_response(page_number, page_path)

    tracker = TokenTracker(state.get("token_usage"))
    tracker.record_usage("process_page_node", input_tokens=10, output_tokens=5)

    logger.info("Mock processed page %s", page_number)
    return {
        "current_response": response.model_dump(mode="json"),
        "token_usage": tracker.to_state(),
    }


def save_page_result_node(state: VisionIndexerState) -> VisionIndexerState:
    run_dir = Path(state["run_dir"])
    debug_memory = bool(state.get("debug_memory", False))
    response = PageProcessingResponse.model_validate(state["current_response"])

    MemoryStore(run_dir, debug_memory=debug_memory).apply_page_memory_update(
        page_number=response.page_index_output.page_number,
        framework_update_md=response.framework_memory_md,
        short_term_update_md=response.short_term_memory_md,
    )
    output_path = PageOutputStore(run_dir).save_page_output(response.page_index_output)

    page_results = list(state.get("page_results", []))
    page_results.append(str(output_path))

    logger.info("Saved page output for page %s", response.page_index_output.page_number)
    return {"page_results": page_results}


def route_next_page_node(state: VisionIndexerState) -> VisionIndexerState:
    next_page_index = int(state.get("current_page_index", 0)) + 1
    page_count = len(state.get("page_image_paths", []))
    next_step = "process_page_node" if next_page_index < page_count else "finalize_run_node"

    return {
        "current_page_index": next_page_index,
        "next_step": next_step,
    }


def select_next_node(state: VisionIndexerState) -> Literal["process_page_node", "finalize_run_node"]:
    next_step = state.get("next_step")
    if next_step not in {"process_page_node", "finalize_run_node"}:
        raise ValueError(f"Invalid next graph step: {next_step}")
    return next_step


def finalize_run_node(state: VisionIndexerState) -> VisionIndexerState:
    run_dir = Path(state["run_dir"])
    token_usage = state.get("token_usage", TokenTracker.empty_state())
    total_tokens = token_usage.get("total", TokenTracker.empty_state()["total"])

    manifest = {
        "run_id": run_dir.name,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_pdf": str(run_dir / "source" / "source.pdf"),
        "page_count": len(state.get("page_image_paths", [])),
        "page_outputs": list(state.get("page_results", [])),
        "debug_memory": bool(state.get("debug_memory", False)),
        "graph_mermaid": str(run_dir / "graph" / "graph.mmd"),
        "token_usage": total_tokens,
    }
    ManifestStore(run_dir).write_manifest(manifest)

    logger.info("Finalized run manifest")
    return {"manifest": manifest}


def _build_mock_page_response(page_number: int, page_path: str) -> PageProcessingResponse:
    section_id = f"section-{page_number:04d}-001"
    return PageProcessingResponse(
        framework_memory_md=f"# Framework Memory\n\nProcessed page {page_number}.\n",
        short_term_memory_md=f"# Short-Term Memory\n\nLatest page: {page_number}.\n",
        page_index_output=PageIndexOutput(
            page_number=page_number,
            page_path=page_path,
            page_type="mock",
            index_worthy=True,
            sections=[
                Section(
                    section_id=section_id,
                    source_section_id=None,
                    heading=f"Mock Section {page_number}",
                    text=f"Mock extracted section for page {page_number}.",
                )
            ],
            titles=[Title(title_id=f"title-{page_number:04d}-001", text=f"Mock Page {page_number}", level=1)],
            topics=[
                Topic(
                    topic_id=None,
                    source_section_id=section_id,
                    label=f"Mock Topic {page_number}",
                    confidence=1.0,
                )
            ],
            assets=[
                Asset(
                    asset_id=f"asset-{page_number:04d}-001",
                    source_section_id=section_id,
                    asset_type="page_image",
                    description=f"Rendered page image for page {page_number}.",
                )
            ],
            brief_summary=f"Mock summary for page {page_number}.",
        ),
    )
