from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Mapping


TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "visible_output_tokens",
    "total_tokens",
)


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
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "visible_output_tokens": 0,
                "total_tokens": 0,
            },
        }

    def record_usage(
        self,
        operation_name: str,
        usage_metadata: Mapping[str, Any] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        page_number: int | None = None,
        provider: str = "",
        model: str = "",
        reasoning_effort: str = "",
        batch_number: int | None = None,
        page_range: str | None = None,
    ) -> None:
        usage = self._normalize_usage(usage_metadata, input_tokens, output_tokens)
        operations = self._state.setdefault("operations", {})
        operation = operations.setdefault(
            operation_name,
            {
                "provider": provider,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "visible_output_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            },
        )
        operation["provider"] = provider or operation.get("provider", "")
        operation["model"] = model or operation.get("model", "")
        operation["reasoning_effort"] = reasoning_effort or operation.get("reasoning_effort", "")

        for field in TOKEN_FIELDS:
            operation[field] += usage[field]
            self._state["total"][field] += usage[field]
        operation["calls"] += 1

        self._logger.info(
            "operation=%s page_number=%s batch_number=%s page_range=%s provider=%s model=%s reasoning_effort=%s input_tokens=%s cached_input_tokens=%s output_tokens=%s reasoning_tokens=%s visible_output_tokens=%s total_tokens=%s",
            operation_name,
            page_number,
            batch_number,
            page_range,
            provider,
            model,
            reasoning_effort,
            usage["input_tokens"],
            usage["cached_input_tokens"],
            usage["output_tokens"],
            usage["reasoning_tokens"],
            usage["visible_output_tokens"],
            usage["total_tokens"],
        )

    def record_langchain_usage(
        self,
        operation_name: str,
        page_number: int | None,
        raw_message: Any,
        provider: str = "",
        model: str = "",
        reasoning_effort: str = "",
        batch_number: int | None = None,
        page_range: str | None = None,
    ) -> None:
        usage_metadata = getattr(raw_message, "usage_metadata", None)
        response_metadata = getattr(raw_message, "response_metadata", None) or {}
        if not usage_metadata:
            usage_metadata = response_metadata.get("token_usage")

        if not usage_metadata:
            self._logger.info(
                "operation=%s page_number=%s batch_number=%s page_range=%s provider=%s model=%s reasoning_effort=%s token_usage_missing=true",
                operation_name,
                page_number,
                batch_number,
                page_range,
                provider,
                model,
                reasoning_effort,
            )

        self.record_usage(
            operation_name,
            usage_metadata=usage_metadata,
            page_number=page_number,
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            batch_number=batch_number,
            page_range=page_range,
        )

    def to_state(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def totals(self) -> dict[str, int]:
        return dict(self._state["total"])

    def log_final_total(self, provider: str = "", model: str = "", reasoning_effort: str = "") -> None:
        total = self.totals()
        self._logger.info(
            "operation=run_total provider=%s model=%s reasoning_effort=%s input_tokens=%s cached_input_tokens=%s output_tokens=%s reasoning_tokens=%s visible_output_tokens=%s total_tokens=%s",
            provider,
            model,
            reasoning_effort,
            total["input_tokens"],
            total["cached_input_tokens"],
            total["output_tokens"],
            total["reasoning_tokens"],
            total["visible_output_tokens"],
            total["total_tokens"],
        )

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
            cached_input_tokens = _nested_int(
                usage_metadata,
                ("input_tokens_details", "cached_tokens"),
                ("prompt_tokens_details", "cached_tokens"),
                ("input_token_details", "cache_read"),
            )
            reasoning_tokens = _nested_int(
                usage_metadata,
                ("output_tokens_details", "reasoning_tokens"),
                ("completion_tokens_details", "reasoning_tokens"),
                ("output_token_details", "reasoning"),
            )
            visible_output_tokens = max(resolved_output - reasoning_tokens, 0)
            return {
                "input_tokens": resolved_input,
                "cached_input_tokens": cached_input_tokens,
                "output_tokens": resolved_output,
                "reasoning_tokens": reasoning_tokens,
                "visible_output_tokens": visible_output_tokens,
                "total_tokens": resolved_total,
            }

        return {
            "input_tokens": input_tokens,
            "cached_input_tokens": 0,
            "output_tokens": output_tokens,
            "reasoning_tokens": 0,
            "visible_output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }


def _nested_int(mapping: Mapping[str, Any], *paths: tuple[str, str]) -> int:
    for first_key, second_key in paths:
        nested = mapping.get(first_key)
        if isinstance(nested, Mapping) and nested.get(second_key) is not None:
            return int(nested[second_key])
    return 0
