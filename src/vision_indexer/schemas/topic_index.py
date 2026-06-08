from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TopicIndexDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    source_pdf: str
    total_pages: int = Field(ge=1)
    document_description: str | None = None


class TopicIndexAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str | None = None
    asset_type: str
    asset_name: str | None = None
    asset_description: str


class TopicIndexPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    page_image_path: str
    description: str
    assets: list[TopicIndexAsset]


class TopicIndexTopic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: str
    topic_name: str
    topic_description: str
    primary_pages: list[TopicIndexPage]


class UnindexedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    page_type: str
    reason: str


class TopicIndexOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document: TopicIndexDocument
    topics: list[TopicIndexTopic]
    unindexed_pages: list[UnindexedPage]
