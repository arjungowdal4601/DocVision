from __future__ import annotations

from pathlib import Path

from vision_indexer.graph.state import VisionIndexerState
from vision_indexer.schemas.run_status import RunStatus
from vision_indexer.storage.csv_checkpoint_store import append_checkpoint_event
from vision_indexer.storage.error_store import write_page_error
from vision_indexer.storage.manifest_store import ManifestStore
from vision_indexer.storage.run_status_store import (
    mark_run_failed,
    save_run_status,
    update_page_status,
)
from vision_indexer.tokenomics.token_tracker import TokenTracker


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
