from __future__ import annotations

from vision_indexer.config import AppConfig
from vision_indexer.llm import provider


def test_build_chat_model_does_not_pass_timeout(monkeypatch) -> None:
    received_kwargs: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            received_kwargs.update(kwargs)

    monkeypatch.setattr(provider, "ChatOpenAI", FakeChatOpenAI)

    model = provider.build_chat_model(
        AppConfig(
            llm_provider="openai",
            model="gpt-test",
            reasoning_effort="high",
            max_retries=4,
            pdf_dpi=150,
            debug_memory=False,
        )
    )

    assert isinstance(model, FakeChatOpenAI)
    assert received_kwargs == {
        "model": "gpt-test",
        "reasoning_effort": "high",
        "max_retries": 4,
    }
