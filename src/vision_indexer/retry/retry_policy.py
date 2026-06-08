from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from os import getenv
from typing import Any, Mapping


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 2.0
    backoff_multiplier: float = 2.0

    @classmethod
    def from_env(cls) -> "RetryPolicy":
        return cls(
            max_attempts=int(getenv("VISION_INDEXER_RETRY_MAX_ATTEMPTS", "3")),
            initial_delay_seconds=float(getenv("VISION_INDEXER_RETRY_INITIAL_DELAY_SECONDS", "2.0")),
            backoff_multiplier=float(getenv("VISION_INDEXER_RETRY_BACKOFF_MULTIPLIER", "2.0")),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RetryPolicy":
        if values is None:
            return cls()
        return cls(
            max_attempts=int(values.get("max_attempts", 3)),
            initial_delay_seconds=float(values.get("initial_delay_seconds", 2.0)),
            backoff_multiplier=float(values.get("backoff_multiplier", 2.0)),
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def sleep_before_retry(attempt: int, policy: RetryPolicy) -> None:
    delay = policy.initial_delay_seconds * (policy.backoff_multiplier ** (attempt - 1))
    time.sleep(delay)
