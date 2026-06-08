import io
import logging
from types import SimpleNamespace

from vision_indexer.tokenomics.token_tracker import TokenTracker


def test_record_langchain_usage_reads_usage_metadata() -> None:
    raw_message = SimpleNamespace(
        usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        response_metadata={},
    )
    tracker = TokenTracker()

    tracker.record_langchain_usage("process_page_node", page_number=1, raw_message=raw_message)

    assert tracker.totals() == {
        "input_tokens": 11,
        "cached_input_tokens": 0,
        "output_tokens": 7,
        "reasoning_tokens": 0,
        "visible_output_tokens": 7,
        "total_tokens": 18,
    }


def test_record_langchain_usage_reads_response_metadata_token_usage() -> None:
    raw_message = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}},
    )
    tracker = TokenTracker()

    tracker.record_langchain_usage("process_page_node", page_number=2, raw_message=raw_message)

    assert tracker.totals() == {
        "input_tokens": 3,
        "cached_input_tokens": 0,
        "output_tokens": 4,
        "reasoning_tokens": 0,
        "visible_output_tokens": 4,
        "total_tokens": 7,
    }


def test_record_langchain_usage_records_zero_when_metadata_missing() -> None:
    raw_message = SimpleNamespace(usage_metadata=None, response_metadata={})
    tracker = TokenTracker()

    tracker.record_langchain_usage("process_page_node", page_number=3, raw_message=raw_message)

    assert tracker.totals() == {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "visible_output_tokens": 0,
        "total_tokens": 0,
    }


def test_record_langchain_usage_reads_openai_native_token_details() -> None:
    raw_message = SimpleNamespace(
        usage_metadata={
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 40},
            "output_tokens": 70,
            "output_tokens_details": {"reasoning_tokens": 25},
            "total_tokens": 170,
        },
        response_metadata={},
    )
    tracker = TokenTracker()

    tracker.record_langchain_usage(
        "process_page_node",
        page_number=4,
        raw_message=raw_message,
        provider="openai",
        model="gpt-5.4",
        reasoning_effort="medium",
    )

    assert tracker.totals() == {
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "output_tokens": 70,
        "reasoning_tokens": 25,
        "visible_output_tokens": 45,
        "total_tokens": 170,
    }


def test_record_langchain_usage_reads_langchain_normalized_token_details() -> None:
    raw_message = SimpleNamespace(
        usage_metadata={
            "input_tokens": 50,
            "input_token_details": {"cache_read": 12},
            "output_tokens": 30,
            "output_token_details": {"reasoning": 8},
            "total_tokens": 80,
        },
        response_metadata={},
    )
    tracker = TokenTracker()

    tracker.record_langchain_usage("process_page_node", page_number=5, raw_message=raw_message)

    assert tracker.totals()["cached_input_tokens"] == 12
    assert tracker.totals()["reasoning_tokens"] == 8
    assert tracker.totals()["visible_output_tokens"] == 22


def test_token_logs_include_model_metadata_and_final_summary() -> None:
    raw_message = SimpleNamespace(
        usage_metadata={
            "input_tokens": 20,
            "input_token_details": {"cache_read": 5},
            "output_tokens": 15,
            "output_token_details": {"reasoning": 6},
            "total_tokens": 35,
        },
        response_metadata={},
    )
    tracker = TokenTracker()
    token_logger = logging.getLogger("vision_indexer.tokenomics")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    token_logger.addHandler(handler)
    old_level = token_logger.level
    token_logger.setLevel(logging.INFO)
    try:
        tracker.record_langchain_usage(
            "process_page_node",
            page_number=6,
            raw_message=raw_message,
            provider="openai",
            model="gpt-5.4",
            reasoning_effort="high",
        )
        tracker.log_final_total(provider="openai", model="gpt-5.4", reasoning_effort="high")
    finally:
        token_logger.removeHandler(handler)
        token_logger.setLevel(old_level)

    log_text = stream.getvalue()
    assert "provider=openai" in log_text
    assert "model=gpt-5.4" in log_text
    assert "reasoning_effort=high" in log_text
    assert "cached_input_tokens=5" in log_text
    assert "reasoning_tokens=6" in log_text
    assert "visible_output_tokens=9" in log_text
    assert "operation=run_total" in log_text


def test_token_logs_include_topic_index_batch_metadata() -> None:
    raw_message = SimpleNamespace(
        usage_metadata={
            "input_tokens": 20,
            "output_tokens": 10,
            "output_token_details": {"reasoning": 4},
            "total_tokens": 30,
        },
        response_metadata={},
    )
    tracker = TokenTracker()
    token_logger = logging.getLogger("vision_indexer.tokenomics")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    token_logger.addHandler(handler)
    old_level = token_logger.level
    token_logger.setLevel(logging.INFO)
    try:
        tracker.record_langchain_usage(
            "build_topic_index_batch",
            page_number=None,
            raw_message=raw_message,
            provider="openai",
            model="gpt-5.4",
            reasoning_effort="medium",
            batch_number=2,
            page_range="11-20",
        )
    finally:
        token_logger.removeHandler(handler)
        token_logger.setLevel(old_level)

    log_text = stream.getvalue()
    assert "operation=build_topic_index_batch" in log_text
    assert "batch_number=2" in log_text
    assert "page_range=11-20" in log_text
    assert "provider=openai" in log_text
    assert "model=gpt-5.4" in log_text
    assert "reasoning_effort=medium" in log_text
    assert tracker.to_state()["operations"]["build_topic_index_batch"]["calls"] == 1
