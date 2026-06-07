from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from vision_indexer.schemas.page_output import PageIndexOutput


class PageProcessingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework_memory_md: str
    short_term_memory_md: str
    page_index_output: PageIndexOutput
