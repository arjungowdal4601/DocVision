from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from vision_indexer.graph.state import VisionIndexerState
from vision_indexer.llm.page_processor import process_page_with_llm
from vision_indexer.llm.topic_indexer import build_topic_index_with_llm
from vision_indexer.memory.memory_store import MemoryStore
from vision_indexer.retry.retry_policy import RetryPolicy, sleep_before_retry
from vision_indexer.schemas.memory_patch import MarkdownMemoryEdit
from vision_indexer.schemas.page_output import PageIndexOutput
from vision_indexer.schemas.run_status import RunStatus
from vision_indexer.storage.csv_checkpoint_store import append_checkpoint_event
from vision_indexer.storage.error_store import write_page_error, write_topic_index_error
from vision_indexer.storage.manifest_store import ManifestStore
from vision_indexer.storage.page_output_store import PageOutputStore
from vision_indexer.storage.run_status_store import (
    initialize_run_status,
    load_run_status,
    mark_run_completed,
    mark_run_failed,
    mark_run_resuming,
    mark_run_running,
    save_run_status,
    update_page_status,
)
from vision_indexer.storage.run_store import RunStore
from vision_indexer.storage.topic_index_store import TopicIndexStore
from vision_indexer.tokenomics.token_tracker import TokenTracker
from vision_indexer.utils.pdf_renderer import render_pdf_to_images


logger = logging.getLogger(__name__)
TOPIC_INDEX_BATCH_SIZE = 10


def initialize_run_node(state: VisionIndexerState) -> VisionIndexerState:
    run_dir = Path(state["run_dir"])
    source_pdf_path = Path(state["source_pdf_path"])
    debug_memory = bool(state.get("debug_memory", False))
    resume = bool(state.get("resume", False))

    run_store = RunStore(run_dir)
    run_store.prepare_directories()
    memory_store = MemoryStore(run_dir, debug_memory=debug_memory)

    if resume:
        run_status = load_run_status(run_dir)
        if run_status is None:
            raise RuntimeError(f"Cannot resume without run_status.json at {run_dir}")
        run_status = mark_run_resuming(run_status)
        save_run_status(run_dir, run_status)
        append_checkpoint_event(
            run_dir,
            event="run_resumed",
            run_id=run_status.run_id,
            status=run_status.status,
            message=f"resume count {run_status.resume_count}",
        )
        copied_pdf_path = run_dir / "source" / "source.pdf"
        if not copied_pdf_path.exists():
            copied_pdf_path = run_store.copy_source_pdf(source_pdf_path)
        logger.info("Resume detected for run at %s", run_dir)
        token_usage = _load_token_usage_seed(run_dir)
    else:
        copied_pdf_path = run_store.copy_source_pdf(source_pdf_path)
        memory_store.initialize()
        run_status = None
        token_usage = TokenTracker.empty_state()

    logger.info("Initialized run at %s", run_dir)
    return {
        "run_source_pdf_path": str(copied_pdf_path),
        "page_image_paths": [],
        "current_page_index": 0,
        "framework_memory_md": memory_store.read_framework_memory(),
        "short_term_memory_md": memory_store.read_short_term_memory(),
        "page_results": _existing_page_results(run_status),
        "completed_pages": 0 if run_status is None else len(run_status.completed_pages),
        "completed_page_numbers": [] if run_status is None else list(run_status.completed_pages),
        "failed_pages": [] if run_status is None else list(run_status.failed_pages),
        "run_status": {} if run_status is None else run_status.model_dump(mode="json"),
        "page_statuses": {} if run_status is None else _page_statuses_payload(run_status),
        "token_usage": token_usage,
    }


def render_pdf_node(state: VisionIndexerState) -> VisionIndexerState:
    run_dir = Path(state["run_dir"])
    pdf_path = Path(state["run_source_pdf_path"])
    dpi = int(state.get("dpi", 150))
    resume = bool(state.get("resume", False))
    force_render = bool(state.get("force_render", False))
    max_pages = state.get("max_pages")

    existing_images = sorted((run_dir / "page_images").glob("page_*.png"))
    if resume and existing_images and not force_render:
        page_image_paths = existing_images
        logger.info("Using %s existing rendered page image(s)", len(page_image_paths))
    else:
        logger.info("Rendering PDF pages")
        page_image_paths = render_pdf_to_images(pdf_path, run_dir / "page_images", dpi=dpi)

    if max_pages is not None:
        page_image_paths = page_image_paths[: int(max_pages)]
    if not page_image_paths:
        raise ValueError(f"PDF contains no pages: {pdf_path}")

    existing_status = _run_status_from_state(state)
    if existing_status is None:
        run_status = initialize_run_status(
            run_id=run_dir.name,
            pdf_path=str(Path(state["source_pdf_path"])),
            output_dir=str(run_dir),
            page_paths=[str(path) for path in page_image_paths],
        )
        checkpoint_event = "run_started"
    else:
        run_status = mark_run_running(_ensure_page_statuses(existing_status, page_image_paths))
        checkpoint_event = None
    save_run_status(run_dir, run_status)
    if checkpoint_event is not None:
        append_checkpoint_event(
            run_dir,
            event=checkpoint_event,
            run_id=run_status.run_id,
            status=run_status.status,
            message=f"{run_status.total_pages} page(s)",
        )

    logger.info("Rendered %s PDF page(s)", len(page_image_paths))
    return {
        "page_image_paths": [str(path) for path in page_image_paths],
        "current_page_index": 0,
        "run_status": run_status.model_dump(mode="json"),
        "page_statuses": _page_statuses_payload(run_status),
        "completed_page_numbers": list(run_status.completed_pages),
        "completed_pages": len(run_status.completed_pages),
        "failed_pages": list(run_status.failed_pages),
    }


def process_page_node(state: VisionIndexerState) -> VisionIndexerState:
    page_image_paths = state["page_image_paths"]
    current_page_index = int(state.get("current_page_index", 0))
    if current_page_index >= len(page_image_paths):
        raise IndexError("Current page index is outside rendered page range")

    page_number = current_page_index + 1
    page_path = Path(page_image_paths[current_page_index])
    framework_memory_md = state["framework_memory_md"]
    short_term_memory_md = state["short_term_memory_md"]
    run_dir = Path(state["run_dir"])
    run_status = _require_run_status(state)

    if _should_skip_completed_page(state, run_status, page_number):
        page_status = run_status.page_statuses[str(page_number)]
        run_status = update_page_status(
            run_status,
            page_number=page_number,
            status="skipped",
            output_path=page_status.output_path,
        )
        save_run_status(run_dir, run_status)
        updated_page_status = run_status.page_statuses[str(page_number)]
        append_checkpoint_event(
            run_dir,
            event="page_skipped",
            run_id=run_status.run_id,
            page_number=page_number,
            status=updated_page_status.status,
            attempts=updated_page_status.attempts,
            output_path=updated_page_status.output_path,
        )
        logger.info("Skipped completed page %s", page_number)
        return {
            "page_action": "skipped",
            "run_status": run_status.model_dump(mode="json"),
            "page_statuses": _page_statuses_payload(run_status),
        }

    logger.info("Starting page %s", page_number)
    retry_policy = RetryPolicy.from_mapping(state.get("retry_policy"))
    last_error: Exception | None = None
    last_error_path: Path | None = None

    for attempt in range(1, retry_policy.max_attempts + 1):
        run_status = update_page_status(
            mark_run_running(run_status),
            page_number=page_number,
            status="running",
            page_path=str(page_path),
            attempts=attempt,
        )
        save_run_status(run_dir, run_status)
        append_checkpoint_event(
            run_dir,
            event="page_started",
            run_id=run_status.run_id,
            page_number=page_number,
            status="running",
            attempts=attempt,
        )
        try:
            logger.info("Page %s attempt %s started", page_number, attempt)
            response, raw_message = process_page_with_llm(
                page_number=page_number,
                page_path=page_path,
                framework_memory_md=framework_memory_md,
                short_term_memory_md=short_term_memory_md,
            )

            tracker = TokenTracker(state.get("token_usage"))
            tracker.record_langchain_usage(
                "process_page_node",
                page_number=page_number,
                raw_message=raw_message,
                provider=str(state.get("llm_provider", "")),
                model=str(state.get("model_name", "")),
                reasoning_effort=str(state.get("reasoning_effort", "")),
            )

            logger.info("Completed page %s", page_number)
            return {
                "page_action": "processed",
                "current_page_output": response.page_index_output.model_dump(mode="json"),
                "pending_framework_memory_edits": [
                    edit.model_dump(mode="json") for edit in response.memory_edits.framework_memory_edits
                ],
                "pending_short_term_memory_edits": [
                    edit.model_dump(mode="json") for edit in response.memory_edits.short_term_memory_edits
                ],
                "run_status": run_status.model_dump(mode="json"),
                "page_statuses": _page_statuses_payload(run_status),
                "token_usage": tracker.to_state(),
            }
        except Exception as exc:
            last_error = exc
            last_error_path = write_page_error(run_dir, page_number=page_number, error=exc, attempt=attempt)
            logger.warning("Page %s attempt %s failed: %s", page_number, attempt, exc)
            if attempt < retry_policy.max_attempts:
                sleep_before_retry(attempt, retry_policy)

    assert last_error is not None
    _persist_page_failure(
        run_dir=run_dir,
        state=state,
        run_status=run_status,
        page_number=page_number,
        error=last_error,
        attempts=retry_policy.max_attempts,
        error_path=last_error_path,
    )
    logger.error("Max retries exceeded for page %s", page_number)
    raise last_error


def save_page_result_node(state: VisionIndexerState) -> VisionIndexerState:
    run_dir = Path(state["run_dir"])
    if state.get("page_action") == "skipped":
        run_status = _require_run_status(state)
        page_number = int(state.get("current_page_index", 0)) + 1
        output_path = run_status.page_statuses[str(page_number)].output_path
        page_results = _append_unique(list(state.get("page_results", [])), output_path)
        return {
            "page_results": page_results,
            "completed_pages": len(run_status.completed_pages),
            "completed_page_numbers": list(run_status.completed_pages),
            "failed_pages": list(run_status.failed_pages),
        }

    debug_memory = bool(state.get("debug_memory", False))
    run_status = _require_run_status(state)
    page_number = _current_page_number(state)

    try:
        page_output = PageIndexOutput.model_validate(state["current_page_output"])
        framework_memory_edits = [
            MarkdownMemoryEdit.model_validate(edit) for edit in state["pending_framework_memory_edits"]
        ]
        short_term_memory_edits = [
            MarkdownMemoryEdit.model_validate(edit) for edit in state["pending_short_term_memory_edits"]
        ]

        memory_store = MemoryStore(run_dir, debug_memory=debug_memory)
        memory_store.apply_page_memory_update(
            page_number=page_output.page_number,
            framework_memory_edits=framework_memory_edits,
            short_term_memory_edits=short_term_memory_edits,
        )
        output_path = PageOutputStore(run_dir).save_page_output(page_output)
        run_status = update_page_status(
            run_status,
            page_number=page_output.page_number,
            status="completed",
            output_path=str(output_path),
        )
        save_run_status(run_dir, run_status)
        page_status = run_status.page_statuses[str(page_output.page_number)]
        append_checkpoint_event(
            run_dir,
            event="page_completed",
            run_id=run_status.run_id,
            page_number=page_output.page_number,
            status=page_status.status,
            attempts=page_status.attempts,
            output_path=page_status.output_path,
        )
    except Exception as exc:
        page_status = run_status.page_statuses.get(str(page_number))
        _persist_page_failure(
            run_dir=run_dir,
            state=state,
            run_status=run_status,
            page_number=page_number,
            error=exc,
            attempts=None if page_status is None else page_status.attempts,
        )
        logger.error("Failed to save page %s result: %s", page_number, exc)
        raise

    page_results = list(state.get("page_results", []))
    page_results = _append_unique(page_results, str(output_path))

    logger.info("Applied memory edits for page %s", page_output.page_number)
    logger.info("Saved page output for page %s", page_output.page_number)
    return {
        "framework_memory_md": memory_store.read_framework_memory(),
        "short_term_memory_md": memory_store.read_short_term_memory(),
        "page_results": page_results,
        "completed_pages": len(run_status.completed_pages),
        "completed_page_numbers": list(run_status.completed_pages),
        "failed_pages": list(run_status.failed_pages),
        "run_status": run_status.model_dump(mode="json"),
        "page_statuses": _page_statuses_payload(run_status),
    }


def route_next_page_node(state: VisionIndexerState) -> VisionIndexerState:
    next_page_index = int(state.get("current_page_index", 0)) + 1
    page_count = len(state.get("page_image_paths", []))
    next_step = "process_page_node" if next_page_index < page_count else "build_topic_index_node"

    return {
        "current_page_index": next_page_index,
        "next_step": next_step,
    }


def select_next_node(state: VisionIndexerState) -> Literal["process_page_node", "build_topic_index_node"]:
    next_step = state.get("next_step")
    if next_step not in {"process_page_node", "build_topic_index_node"}:
        raise ValueError(f"Invalid next graph step: {next_step}")
    return next_step


def build_topic_index_node(state: VisionIndexerState) -> VisionIndexerState:
    run_dir = Path(state["run_dir"])
    run_status = _require_run_status(state)
    topic_index_store = TopicIndexStore(run_dir)
    failed_batch_number: int | None = None
    failed_page_range: str | None = None

    try:
        page_outputs = topic_index_store.load_page_outputs()
        if not page_outputs:
            raise ValueError("Cannot build topic index because no page outputs were found.")

        framework_memory_md = MemoryStore(run_dir).read_framework_memory()
        tracker = TokenTracker(state.get("token_usage"))
        previous_topic_index = None
        total_batches = (len(page_outputs) + TOPIC_INDEX_BATCH_SIZE - 1) // TOPIC_INDEX_BATCH_SIZE
        logger.info(
            "Starting topic indexing for %s page output(s) in %s batch(es)",
            len(page_outputs),
            total_batches,
        )

        for batch_number, start_index in enumerate(range(0, len(page_outputs), TOPIC_INDEX_BATCH_SIZE), start=1):
            batch_page_outputs = page_outputs[start_index : start_index + TOPIC_INDEX_BATCH_SIZE]
            start_page = batch_page_outputs[0].page_number
            end_page = batch_page_outputs[-1].page_number
            page_range = f"{start_page}-{end_page}"
            failed_batch_number = batch_number
            failed_page_range = page_range
            logger.info(
                "Topic index batch %s/%s started for pages %s",
                batch_number,
                total_batches,
                page_range,
            )
            topic_index, raw_message = build_topic_index_with_llm(
                page_outputs=batch_page_outputs,
                source_pdf_path=Path(state["source_pdf_path"]),
                previous_topic_index=previous_topic_index,
                framework_memory_md=framework_memory_md,
                batch_number=batch_number,
                total_batches=total_batches,
            )
            batch_path = topic_index_store.save_batch_topic_index(batch_number, topic_index)
            tracker.record_langchain_usage(
                "build_topic_index_batch",
                page_number=None,
                raw_message=raw_message,
                provider=str(state.get("llm_provider", "")),
                model=str(state.get("model_name", "")),
                reasoning_effort=str(state.get("reasoning_effort", "")),
                batch_number=batch_number,
                page_range=page_range,
            )
            previous_topic_index = topic_index
            logger.info("Saved topic index batch %s at %s", batch_number, batch_path)

        assert previous_topic_index is not None
        topic_index_path = topic_index_store.save_topic_index(previous_topic_index)

        logger.info("Saved final topic index at %s", topic_index_path)
        return {
            "topic_index_path": str(topic_index_path),
            "token_usage": tracker.to_state(),
        }
    except Exception as exc:
        error_path = write_topic_index_error(
            run_dir,
            exc,
            batch_number=failed_batch_number,
            page_range=failed_page_range,
        )
        failed_status = mark_run_failed(run_status, current_page=None)
        save_run_status(run_dir, failed_status)
        append_checkpoint_event(
            run_dir,
            event="topic_index_failed",
            run_id=failed_status.run_id,
            status="failed",
            error_path=str(error_path),
            message=str(exc),
        )
        _write_manifest(run_dir, failed_status, state)
        logger.error("Failed to build topic index: %s", exc)
        raise


def finalize_run_node(state: VisionIndexerState) -> VisionIndexerState:
    run_dir = Path(state["run_dir"])
    run_status = mark_run_completed(_require_run_status(state))
    save_run_status(run_dir, run_status)
    append_checkpoint_event(
        run_dir,
        event="run_completed",
        run_id=run_status.run_id,
        status=run_status.status,
        message=f"{len(run_status.completed_pages)} completed page(s)",
    )
    token_usage = state.get("token_usage", TokenTracker.empty_state())
    tracker = TokenTracker(token_usage)
    tracker.log_final_total(
        provider=str(state.get("llm_provider", "")),
        model=str(state.get("model_name", "")),
        reasoning_effort=str(state.get("reasoning_effort", "")),
    )
    manifest_path = _write_manifest(run_dir, run_status, state)

    logger.info("Finalized run manifest")
    return {
        "manifest": ManifestStore(run_dir).load_manifest() or {"manifest_path": str(manifest_path)},
        "run_status": run_status.model_dump(mode="json"),
        "page_statuses": _page_statuses_payload(run_status),
    }


def _run_status_from_state(state: VisionIndexerState) -> RunStatus | None:
    raw_status = state.get("run_status")
    if not raw_status:
        return None
    return RunStatus.model_validate(raw_status)


def _require_run_status(state: VisionIndexerState) -> RunStatus:
    run_status = _run_status_from_state(state)
    if run_status is None:
        raise RuntimeError("Run status is not initialized")
    return run_status


def _ensure_page_statuses(run_status: RunStatus, page_image_paths: list[Path]) -> RunStatus:
    updated = run_status.model_copy(deep=True)
    updated.total_pages = len(page_image_paths)
    for index, page_path in enumerate(page_image_paths, start=1):
        key = str(index)
        if key not in updated.page_statuses:
            updated.page_statuses[key] = update_page_status(
                updated,
                page_number=index,
                status="pending",
                page_path=str(page_path),
            ).page_statuses[key]
        else:
            updated.page_statuses[key].page_path = str(page_path)
    return updated


def _page_statuses_payload(run_status: RunStatus) -> dict[str, dict]:
    return {key: value.model_dump(mode="json") for key, value in run_status.page_statuses.items()}


def _should_skip_completed_page(state: VisionIndexerState, run_status: RunStatus, page_number: int) -> bool:
    if bool(state.get("force", False)):
        return False
    if page_number in set(state.get("force_pages", [])):
        return False
    page_status = run_status.page_statuses.get(str(page_number))
    return page_number in run_status.completed_pages and page_status is not None and page_status.output_path is not None


def _existing_page_results(run_status: RunStatus | None) -> list[str]:
    if run_status is None:
        return []
    results: list[str] = []
    for page_number in run_status.completed_pages:
        page_status = run_status.page_statuses.get(str(page_number))
        if page_status and page_status.output_path:
            results.append(page_status.output_path)
    return results


def _append_unique(values: list[str], value: str | None) -> list[str]:
    if value and value not in values:
        values.append(value)
    return values


def _current_page_number(state: VisionIndexerState) -> int:
    raw_output = state.get("current_page_output")
    if isinstance(raw_output, dict) and raw_output.get("page_number") is not None:
        return int(raw_output["page_number"])
    return int(state.get("current_page_index", 0)) + 1


def _persist_page_failure(
    *,
    run_dir: Path,
    state: VisionIndexerState,
    run_status: RunStatus,
    page_number: int,
    error: Exception,
    attempts: int | None,
    error_path: Path | None = None,
) -> RunStatus:
    resolved_error_path = error_path or write_page_error(
        run_dir,
        page_number=page_number,
        error=error,
        attempt=attempts,
    )
    failed_status = update_page_status(
        run_status,
        page_number=page_number,
        status="failed",
        error_message=str(error),
        attempts=attempts,
    )
    failed_status = mark_run_failed(failed_status, current_page=page_number)
    save_run_status(run_dir, failed_status)
    append_checkpoint_event(
        run_dir,
        event="page_failed",
        run_id=failed_status.run_id,
        page_number=page_number,
        status="failed",
        attempts=attempts,
        error_path=str(resolved_error_path),
        message=str(error),
    )
    _write_manifest(run_dir, failed_status, state)
    return failed_status


def _load_token_usage_seed(run_dir: Path) -> dict:
    manifest = ManifestStore(run_dir).load_manifest()
    if not manifest:
        return TokenTracker.empty_state()
    totals = manifest.get("token_totals") or manifest.get("token_usage")
    if not totals:
        return TokenTracker.empty_state()
    return {"operations": {}, "total": totals}


def _write_manifest(run_dir: Path, run_status: RunStatus, state: VisionIndexerState) -> Path:
    token_usage = state.get("token_usage", TokenTracker.empty_state())
    token_totals = token_usage.get("total", TokenTracker.empty_state()["total"])
    return ManifestStore(run_dir).write_run_manifest(
        run_status=run_status,
        token_totals=token_totals,
        debug_mode=bool(state.get("debug_memory", False)),
        model_name=str(state.get("model_name", "")),
        reasoning_effort=str(state.get("reasoning_effort", "")),
    )
