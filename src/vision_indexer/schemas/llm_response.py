from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from vision_indexer.schemas.memory_patch import MemoryEditBundle
from vision_indexer.schemas.page_output import PageIndexOutput


class PageProcessingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_edits: MemoryEditBundle
    page_index_output: PageIndexOutput
