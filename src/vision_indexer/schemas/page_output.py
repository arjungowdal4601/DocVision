from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    source_section_id: str | None = None
    heading: str
    text: str


class Title(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title_id: str
    text: str
    level: int | None = None


class Topic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: str | None = None
    source_section_id: str | None = None
    label: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    source_section_id: str | None = None
    asset_type: str
    description: str


class PageIndexOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    page_path: str
    page_type: str
    index_worthy: bool
    sections: list[Section]
    titles: list[Title]
    topics: list[Topic]
    assets: list[Asset]
    brief_summary: str
