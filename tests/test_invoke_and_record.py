"""Unit tests for the provider-agnostic behavior of ``claude._invoke_and_record``
(config#1659 Phase B re-key).

Two behaviors added on top of the existing Anthropic-only contract:

1. A transport that cannot force its server-side search tool via
   ``tool_choice`` (``SearchOptions.force_first`` raises ``LLMConfigError`` —
   see ``krepis.llm``, the OpenRouter ``openrouter:web_search`` server tool)
   degrades to a second, unforced call rather than propagating the error —
   the recovery pass's prose directive still asks for the search, it just
   isn't hard-guaranteed on that transport.
2. The ``min_web_searches`` floor's search count is read from
   ``max(len(result.searches), result.usage.web_search_requests)`` so a
   transport that never populates per-query events (OpenRouter) still
   reports the real count via the normalized ``usage`` field instead of a
   false zero.

Uses real ``krepis.llm`` dataclasses (``GroundedResult``, ``LLMUsage``,
``ModelSpec``) with a duck-typed fake client — no network, no real SDK.
"""

from __future__ import annotations

import pytest
from krepis.llm import GroundedResult, LLMUsage
from krepis.llm_config import LLMConfigError, ModelSpec

from morning_signal import claude


@pytest.fixture
def patched(monkeypatch):
    """Stub the telemetry sinks so _invoke_and_record's side effects never
    touch disk — mirrors the fixture in test_claude_recovery.py."""
    monkeypatch.setattr(claude, "record_result_cost", lambda **kw: 0.0)
    monkeypatch.setattr(
        claude, "record_search_events", lambda **kw: len(kw["searches"])
    )
    monkeypatch.setattr(claude, "capture_llm_call", lambda *a, **kw: False)


class _FakeLLMClient:
    """Duck-typed stand-in for ``krepis.llm.LLMClient`` — a ``spec`` plus a
    scripted ``complete_grounded`` / ``complete`` that returns/raises from a
    fixed plan. ``complete`` is used by the litellm (and other non-web-search)
    path when ``complete_grounded`` raises ``LLMConfigError``.
    """

    def __init__(self, spec: ModelSpec, plan: list, complete_plan: list | None = None):
        self.spec = spec
        self._plan = list(plan)
        self._complete_plan = list(complete_plan or [])
        self.force_first_calls: list[bool] = []
        self.complete_calls: int = 0

    def complete_grounded(self, *, system, user_content, search, max_tokens, cache_system):
        self.force_first_calls.append(search.force_first)
        if not self._plan:
            raise AssertionError("complete_grounded called more times than scripted")
        step = self._plan.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def complete(self, *, system, user_content, max_tokens, cache_system, on_unsupported="raise"):
        self.complete_calls += 1
        if not self._complete_plan:
            raise AssertionError("complete called more times than scripted")
        step = self._complete_plan.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _grounded(*, provider, searches=(), citations=(), web_search_requests=0):
    return GroundedResult(
        text="ok",
        model="m",
        provider=provider,
        usage=LLMUsage(web_search_requests=web_search_requests),
        raw_request={},
        raw_response=None,
        searches=list(searches),
        citations=list(citations),
    )


def test_force_first_unsupported_falls_back_to_unforced_retry(patched):
    good = _grounded(
        provider="openrouter",
        citations=[{"url": "https://x", "title": "maga rally", "snippet": None}],
        web_search_requests=3,
    )
    client = _FakeLLMClient(
        ModelSpec("openrouter", "moonshotai/kimi-k2.6"),
        [LLMConfigError("force_first not supported on openrouter"), good],
    )

    result, n_searches = claude._invoke_and_record(
        client, {}, "sys", "user", "2026-07-06", "am", force_search=True,
    )

    assert result is good
    # First attempt forced, retry unforced — not silently skipped.
    assert client.force_first_calls == [True, False]
    # Provider-agnostic count comes from usage.web_search_requests since
    # searches (per-query telemetry) is empty on this transport.
    assert n_searches == 3


def test_llmconfigerror_propagates_when_not_forcing(patched):
    client = _FakeLLMClient(
        ModelSpec("openrouter", "moonshotai/kimi-k2.6"),
        [LLMConfigError("some unrelated config problem")],
    )
    with pytest.raises(LLMConfigError):
        claude._invoke_and_record(
            client, {}, "sys", "user", "2026-07-06", "am", force_search=False,
        )


def test_n_searches_uses_max_of_recorded_and_usage_on_anthropic(patched):
    # Anthropic populates BOTH result.searches and usage.web_search_requests;
    # when usage under-reports (or is absent, as in duck-typed test fakes
    # elsewhere in this suite) the per-query count still wins.
    result = _grounded(
        provider="anthropic",
        searches=[{"query": "q", "urls": [], "result_count": 0, "error": None}],
        web_search_requests=0,
    )
    client = _FakeLLMClient(ModelSpec("anthropic", "claude-haiku-4-5"), [result])

    _, n_searches = claude._invoke_and_record(
        client, {}, "sys", "user", "2026-07-06", "am",
    )

    assert n_searches == 1
    assert client.force_first_calls == [False]


def test_n_searches_falls_back_to_usage_when_searches_empty(patched):
    # OpenRouter shape: searches always empty, usage carries the real count.
    result = _grounded(provider="openrouter", searches=[], web_search_requests=5)
    client = _FakeLLMClient(ModelSpec("openrouter", "moonshotai/kimi-k2.6"), [result])

    _, n_searches = claude._invoke_and_record(
        client, {}, "sys", "user", "2026-07-06", "am",
    )

    assert n_searches == 5


def test_router_edge_falls_through_to_complete_when_grounded_unsupported(patched):
    """The router edge (and any non-anthropic/openrouter provider) has no
    server-side web_search — complete_grounded raises LLMConfigError and the
    path falls through to complete(), wrapping the plain text result as a
    GroundedResult with empty searches/citations. Grounding then comes from the
    pre-fetched news_context digest injected by build_episode_request.

    Was written against `ModelSpec("litellm", ...)`, the IN-PROCESS Router.
    krepis refuses to construct that spec at all, and production no longer
    builds one — `litellm_proxy` is what a resolved router group looks like.
    """
    from types import SimpleNamespace

    complete_result = SimpleNamespace(
        text="Welcome to Morning Signal. News-context grounded.",
        model="deepseek-v4-pro",
        provider="litellm_proxy",
        usage=LLMUsage(),
        raw_request={},
        raw_response=None,
    )
    client = _FakeLLMClient(
        ModelSpec("litellm_proxy", "high", base_url="https://router.example:8443", api_key_env="ROUTER_CONSUMER_TEST"),
        plan=[LLMConfigError("complete_grounded unsupported on litellm_proxy")],
        complete_plan=[complete_result],
    )

    result, n_searches = claude._invoke_and_record(
        client, {}, "sys", "user", "2026-08-02", "am", force_search=False,
    )

    assert result.text == "Welcome to Morning Signal. News-context grounded."
    assert result.provider == "litellm_proxy"
    assert result.model == "deepseek-v4-pro"
    assert result.searches == []
    assert result.citations == []
    assert n_searches == 0
    assert client.complete_calls == 1
    assert client.force_first_calls == [False]


def test_router_edge_force_search_is_noop_via_complete(patched):
    """force_search=True on a non-web-search transport must not raise —
    force is a no-op and complete() still produces the script.
    """
    from types import SimpleNamespace

    complete_result = SimpleNamespace(
        text="ok",
        model="deepseek-v4-pro",
        provider="litellm_proxy",
        usage=LLMUsage(),
        raw_request={},
        raw_response=None,
    )
    client = _FakeLLMClient(
        ModelSpec("litellm_proxy", "high", base_url="https://router.example:8443", api_key_env="ROUTER_CONSUMER_TEST"),
        plan=[LLMConfigError("complete_grounded unsupported on litellm_proxy")],
        complete_plan=[complete_result],
    )

    result, n_searches = claude._invoke_and_record(
        client, {}, "sys", "user", "2026-08-02", "am", force_search=True,
    )

    assert result.text == "ok"
    assert n_searches == 0
    assert client.complete_calls == 1
