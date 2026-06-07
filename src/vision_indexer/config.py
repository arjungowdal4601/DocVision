from __future__ import annotations

from dataclasses import dataclass
from os import getenv

from dotenv import load_dotenv


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str = "openai"
    model: str = "gpt-5.4"
    reasoning_effort: str = "medium"
    timeout_seconds: int = 60
    max_retries: int = 2
    pdf_dpi: int = 150
    debug_memory: bool = False

    @classmethod
    def from_env(cls, debug_memory: bool | None = None) -> "AppConfig":
        load_dotenv()
        env_debug = _parse_bool(getenv("DEBUG_MEMORY"), default=False)
        return cls(
            llm_provider=getenv("VISION_INDEXER_LLM_PROVIDER", "openai"),
            model=getenv("VISION_INDEXER_MODEL", "gpt-5.4"),
            reasoning_effort=getenv("VISION_INDEXER_REASONING_EFFORT", "medium"),
            timeout_seconds=_parse_int(getenv("VISION_INDEXER_TIMEOUT_SECONDS"), 60),
            max_retries=_parse_int(getenv("VISION_INDEXER_MAX_RETRIES"), 2),
            pdf_dpi=_parse_int(getenv("VISION_INDEXER_PDF_DPI"), 150),
            debug_memory=env_debug if debug_memory is None else debug_memory,
        )
