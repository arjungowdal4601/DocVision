from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from vision_indexer.config import AppConfig


def build_chat_model(config: AppConfig | None = None) -> BaseChatModel:
    resolved_config = config or AppConfig.from_env()

    if resolved_config.llm_provider.lower() != "openai":
        raise ValueError(f"Unsupported LLM provider: {resolved_config.llm_provider}")

    return ChatOpenAI(
        model=resolved_config.model,
        reasoning_effort=resolved_config.reasoning_effort,
        max_retries=resolved_config.max_retries,
    )
