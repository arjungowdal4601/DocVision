from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision_indexer.schemas.run_status import RunStatus


class ManifestStore:
    def __init__(self, run_dir: Path) -> None:
        self.manifest_path = run_dir / "manifest.json"

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return self.manifest_path

    def load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write_run_manifest(
        self,
        run_status: RunStatus,
        token_totals: dict[str, Any],
        debug_mode: bool,
        model_name: str,
        reasoning_effort: str,
    ) -> Path:
        manifest = {
            "run_id": run_status.run_id,
            "pdf_path": run_status.pdf_path,
            "output_dir": run_status.output_dir,
            "status": run_status.status,
            "total_pages": run_status.total_pages,
            "page_count": run_status.total_pages,
            "completed_page_count": len(run_status.completed_pages),
            "failed_page_count": len(run_status.failed_pages),
            "completed_pages": run_status.completed_pages,
            "failed_pages": run_status.failed_pages,
            "started_at": run_status.started_at,
            "finished_at": run_status.finished_at,
            "token_totals": token_totals,
            "resume_count": run_status.resume_count,
            "debug_mode": debug_mode,
            "model_name": model_name,
            "reasoning_effort": reasoning_effort,
        }
        return self.write_manifest(manifest)
