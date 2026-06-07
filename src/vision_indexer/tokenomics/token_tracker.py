from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Mapping


TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


class TokenTracker:
    def __init__(self, state: Mapping[str, Any] | None = None) -> None:
        self._state: dict[str, Any] = deepcopy(dict(state)) if state else self.empty_state()
        self._logger = logging.getLogger("vision_indexer.tokenomics")

    @staticmethod
    def empty_state() -> dict[str, Any]:
        return {
            "operations": {},
            "total": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }

    def record_usage(
        self,
        operation_name: str,
        usage_metadata: Mapping[str, Any] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        usage = self._normalize_usage(usage_metadata, input_tokens, output_tokens)
        operations = self._state.setdefault("operations", {})
        operation = operations.setdefault(
            operation_name,
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0},
        )

        for field in TOKEN_FIELDS:
            operation[field] += usage[field]
            self._state["total"][field] += usage[field]
        operation["calls"] += 1

        self._logger.info(
            "operation=%s input_tokens=%s output_tokens=%s total_tokens=%s",
            operation_name,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["total_tokens"],
        )

    def to_state(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def totals(self) -> dict[str, int]:
        return dict(self._state["total"])

    @staticmethod
    def _normalize_usage(
        usage_metadata: Mapping[str, Any] | None,
        input_tokens: int,
        output_tokens: int,
    ) -> dict[str, int]:
        if usage_metadata:
            resolved_input = int(
                usage_metadata.get("input_tokens")
                or usage_metadata.get("prompt_tokens")
                or usage_metadata.get("input_token_count")
                or 0
            )
            resolved_output = int(
                usage_metadata.get("output_tokens")
                or usage_metadata.get("completion_tokens")
                or usage_metadata.get("output_token_count")
                or 0
            )
            resolved_total = int(usage_metadata.get("total_tokens") or resolved_input + resolved_output)
            return {
                "input_tokens": resolved_input,
                "output_tokens": resolved_output,
                "total_tokens": resolved_total,
            }

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
