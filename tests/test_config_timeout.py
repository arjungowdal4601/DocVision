from __future__ import annotations

from pathlib import Path

from vision_indexer.config import AppConfig


def test_app_config_has_no_timeout_seconds() -> None:
    config = AppConfig()

    assert not hasattr(config, "timeout_seconds")


def test_env_example_does_not_document_timeout() -> None:
    env_example = Path(".env.example").read_text(encoding="utf-8")

    assert "VISION_INDEXER_TIMEOUT_SECONDS" not in env_example
