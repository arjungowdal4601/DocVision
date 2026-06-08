from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Topic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: str | None = None
    topic_name: str
    topic_description: str


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str | None = None
    asset_type: str
    asset_name: str | None = None
    asset_description: str


class PageIndexOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    page_type: str
    page_image_path: str
    index_worthy: bool
    summary: str
    topics: list[Topic]
    assets: list[Asset]
