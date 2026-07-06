"""Resilience tests for the shared LLM retry policy (llm_utils.generate_with_retry)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

import llm_utils
from llm_utils import LLMUnavailableError, generate_with_retry


class _FakeModels:
    def __init__(self, behavior):  # type: ignore[no-untyped-def]
        self._behavior = behavior
        self.calls = 0

    async def generate_content(self, **_kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self._behavior(self.calls)


class _FakeClient:
    def __init__(self, behavior):  # type: ignore[no-untyped-def]
        self.aio = type("Aio", (), {"models": _FakeModels(behavior)})()


class _Resp:
    text = "Accept."


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch asyncio.sleep so backoff waits are instant."""
    async def _instant(_seconds):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(llm_utils.asyncio, "sleep", _instant)


def _call(client):  # type: ignore[no-untyped-def]
    return asyncio.run(
        generate_with_retry(client, model="m", contents="c", config=None, source="test")
    )


def test_transport_error_retries_then_escalates() -> None:
    """httpx transport errors are retried, then escalate to LLMUnavailableError."""
    def always_drop(_n):  # type: ignore[no-untyped-def]
        raise httpx.ReadError("connection dropped")

    client = _FakeClient(always_drop)
    with pytest.raises(LLMUnavailableError):
        _call(client)
    assert client.aio.models.calls == llm_utils.LLM_MAX_RETRIES


def test_transport_error_recovers_on_retry() -> None:
    """A transient drop that clears on a later attempt returns the response."""
    def drop_then_ok(n):  # type: ignore[no-untyped-def]
        if n == 1:
            raise httpx.ReadError("transient")
        return _Resp()

    client = _FakeClient(drop_then_ok)
    resp = _call(client)
    assert resp.text == "Accept."
    assert client.aio.models.calls == 2


def test_daily_quota_not_retried() -> None:
    """A per-DAY quota 429 escalates immediately with no retry."""
    from google.genai import errors as genai_errors

    def daily_quota(_n):  # type: ignore[no-untyped-def]
        raise genai_errors.ClientError(429, {"error": {"message": "PerDay quota exceeded"}})

    client = _FakeClient(daily_quota)
    with pytest.raises(LLMUnavailableError):
        _call(client)
    assert client.aio.models.calls == 1
