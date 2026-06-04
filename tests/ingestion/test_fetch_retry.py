"""Retry-path tests for ``ingestion.entsoe.fetch_entsoe_data``.

The network is fully mocked: ``ingestion.entsoe.requests.get`` is replaced by
a fake that yields a scripted sequence of responses, and
``ingestion.entsoe.time.sleep`` is monkeypatched to a no-op so the 30/60/90s
backoff never actually blocks the test. Nothing here hits the real API or
uses a real token.
"""

import requests
import pytest

import ingestion.entsoe as entsoe


class _FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Neutralise backoff so retries don't block the test."""
    sleeps = []
    monkeypatch.setattr(entsoe.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _install_get(monkeypatch, responses):
    """Patch requests.get to return the queued responses in order.

    Returns the list of (args, kwargs) it was called with, so the test can
    assert the security token was injected into params.
    """
    calls = []
    queue = list(responses)

    def fake_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return queue.pop(0)

    monkeypatch.setattr(entsoe.requests, "get", fake_get)
    return calls


def test_retries_on_504_then_succeeds(monkeypatch, _no_sleep):
    calls = _install_get(
        monkeypatch,
        [_FakeResponse(504, "gateway timeout"), _FakeResponse(200, "<xml>ok</xml>")],
    )

    result = entsoe.fetch_entsoe_data({"documentType": "A44"})

    assert result == "<xml>ok</xml>"
    assert len(calls) == 2  # one failed attempt + one success
    assert len(_no_sleep) == 1  # exactly one backoff sleep before the retry
    assert _no_sleep[0] == 30  # 30 * (attempt + 1) for attempt 0
    # securityToken is injected into params by fetch_entsoe_data
    assert "securityToken" in calls[0]["params"]


@pytest.mark.parametrize("status", [502, 503, 504])
def test_retries_on_all_gateway_codes(monkeypatch, _no_sleep, status):
    calls = _install_get(
        monkeypatch,
        [_FakeResponse(status), _FakeResponse(200, "<ok/>")],
    )
    result = entsoe.fetch_entsoe_data({"documentType": "A65"})
    assert result == "<ok/>"
    assert len(calls) == 2


def test_exhausts_retries_and_returns_none(monkeypatch, _no_sleep):
    # All three attempts return 504 -> returns None, sleeps twice (after
    # attempts 0 and 1; the final attempt does not sleep).
    calls = _install_get(
        monkeypatch,
        [_FakeResponse(504), _FakeResponse(504), _FakeResponse(504)],
    )
    result = entsoe.fetch_entsoe_data({"documentType": "A75"}, max_retries=3)
    assert result is None
    assert len(calls) == 3
    assert _no_sleep == [30, 60]  # increasing backoff, no sleep after last try


def test_non_retryable_error_returns_none_immediately(monkeypatch, _no_sleep):
    # A 400 is not in (504, 503, 502) -> no retry, immediate None.
    calls = _install_get(monkeypatch, [_FakeResponse(400, "Bad Request")])
    result = entsoe.fetch_entsoe_data({"documentType": "A44"})
    assert result is None
    assert len(calls) == 1
    assert _no_sleep == []  # never slept


def test_network_exception_is_retried(monkeypatch, _no_sleep):
    # Connectivity errors are caught and retried, then succeed.
    attempts = {"n": 0}

    def flaky_get(url, params=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise requests.exceptions.ConnectionError("boom")
        return _FakeResponse(200, "<recovered/>")

    monkeypatch.setattr(entsoe.requests, "get", flaky_get)

    result = entsoe.fetch_entsoe_data({"documentType": "A44"})
    assert result == "<recovered/>"
    assert attempts["n"] == 2
    assert len(_no_sleep) == 1


def test_first_attempt_success_does_not_sleep(monkeypatch, _no_sleep):
    calls = _install_get(monkeypatch, [_FakeResponse(200, "<first/>")])
    result = entsoe.fetch_entsoe_data({"documentType": "A44"})
    assert result == "<first/>"
    assert len(calls) == 1
    assert _no_sleep == []
