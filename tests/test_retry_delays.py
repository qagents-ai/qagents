from qagents.models._retry import retry_after_seconds


def test_retry_after_http_date_uses_server_minimum(monkeypatch):
    monkeypatch.setattr("qagents.models._retry.time.time", lambda: 784111777)
    assert retry_after_seconds({"retry-after": "Sun, 06 Nov 1994 08:51:37 GMT"}) == 120
    assert retry_after_seconds({"retry-after": "Sun, 06 Nov 1994 08:01:37 GMT"}) == 0


def test_millisecond_precedence_and_invalid_delay_fallback():
    assert retry_after_seconds({"retry-after-ms": "1250", "retry-after": "30"}) == 1.25
    assert retry_after_seconds({"retry-after-ms": "inf", "retry-after": "90"}) == 90
    for text in ("", "-1", "nan", "infinity", "unparseable"):
        assert retry_after_seconds({"retry-after": text}) is None
