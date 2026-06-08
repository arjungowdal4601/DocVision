from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


MemoryEditType = Literal["no_change", "append_new_section", "append_to_section", "replace_section"]


class MarkdownMemoryEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edit_type: MemoryEditType
    section_heading: str | None = None
    content_md: str = ""


class MemoryEditBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework_memory_edits: list[MarkdownMemoryEdit]
    short_term_memory_edits: list[MarkdownMemoryEdit]
