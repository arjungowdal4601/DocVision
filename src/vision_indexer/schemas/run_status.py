from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


PageStatusValue = Literal["pending", "running", "completed", "failed", "skipped"]
RunStatusValue = Literal["not_started", "running", "completed", "failed", "resuming"]


class PageStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    page_path: str
    status: PageStatusValue = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    error_message: str | None = None
    attempts: int = 0
    output_path: str | None = None


class RunStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    pdf_path: str
    output_dir: str
    status: RunStatusValue = "not_started"
    total_pages: int
    completed_pages: list[int] = Field(default_factory=list)
    failed_pages: list[int] = Field(default_factory=list)
    current_page: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    page_statuses: dict[str, PageStatus] = Field(default_factory=dict)
    resume_count: int = 0
