from vision_indexer.retry.retry_policy import RetryPolicy, sleep_before_retry


def test_sleep_before_retry_uses_exponential_delay(monkeypatch) -> None:
    delays: list[float] = []

    monkeypatch.setattr("vision_indexer.retry.retry_policy.time.sleep", delays.append)

    sleep_before_retry(attempt=1, policy=RetryPolicy(initial_delay_seconds=0.5, backoff_multiplier=3.0))
    sleep_before_retry(attempt=2, policy=RetryPolicy(initial_delay_seconds=0.5, backoff_multiplier=3.0))

    assert delays == [0.5, 1.5]
