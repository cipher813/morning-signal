"""Tests for the Kimi-primary / Anthropic-fallback cascade (config#1659).

When the primary ``llm`` spec is non-Anthropic (e.g. an OpenRouter
open-weight model) and its own attempt (first pass + self-heal recovery)
either hard-fails (min_web_searches floor) or silently produces no usable
content — a real, live-verified failure mode for reasoning-capable
OpenRouter models, 2026-07-06 — ``generate_script`` falls through to ONE
fresh attempt on the Anthropic default rather than aborting outright.
Every call also writes a ``{date}-{edition}.llm_decision.json`` recording
which model actually produced (or failed to produce) the script.

Uses a duck-typed fake ``LLMClient`` (real krepis ``GroundedResult``/
``LLMUsage`` dataclasses, no real SDK/network) dispatched by provider, so
each test scripts exactly what the primary vs. fallback calls return.
"""

from __future__ import annotations

import json

import pytest
from krepis.llm import GroundedResult, LLMUsage
from krepis.llm_config import ModelSpec

from morning_signal import claude


def _grounded(*, provider, model, text, n_searches, citations=None):
    if citations is None:
        citations = [{"url": "https://example.com/news", "title": "Example News Article"}]
    return GroundedResult(
        text=text, model=model, provider=provider,
        usage=LLMUsage(web_search_requests=n_searches),
        raw_request={}, raw_response=None, searches=[], citations=citations,
    )


def _client_factory(plan):
    """``plan`` maps provider -> list of GroundedResult to pop per call
    (so a provider hit twice, e.g. primary first-pass + recovery, can
    return different results in sequence).

    Providers without server-side web search (``litellm``) never get a
    successful ``complete_grounded`` — they raise ``LLMConfigError`` and
    the production path falls through to ``complete()``. Script those
    providers' results the same way; ``complete`` returns a SimpleNamespace
    shaped like ``LLMResult`` so the wrap-to-GroundedResult path works.
    """
    from types import SimpleNamespace

    from krepis.llm_config import LLMConfigError

    remaining = {k: list(v) for k, v in plan.items()}

    class _FakeClient:
        def __init__(self, spec, **kw):
            self.spec = spec

        def complete_grounded(self, **kw):
            if self.spec.provider not in ("anthropic", "openrouter"):
                raise LLMConfigError(
                    f"complete_grounded unsupported on {self.spec.provider}"
                )
            queue = remaining.get(self.spec.provider)
            if not queue:
                raise AssertionError(
                    f"no more scripted responses for provider={self.spec.provider!r}"
                )
            return queue.pop(0)

        def complete(self, **kw):
            queue = remaining.get(self.spec.provider)
            if not queue:
                raise AssertionError(
                    f"no more scripted complete() responses for "
                    f"provider={self.spec.provider!r}"
                )
            gr = queue.pop(0)
            # Accept either a GroundedResult (tests script the final shape)
            # or a SimpleNamespace already shaped like LLMResult.
            if isinstance(gr, SimpleNamespace):
                return gr
            return SimpleNamespace(
                text=gr.text,
                model=gr.model,
                provider=gr.provider,
                usage=gr.usage,
                raw_request=gr.raw_request,
                raw_response=gr.raw_response,
            )

    return _FakeClient


def _client_factory_with_capture(plan):
    """Like :func:`_client_factory` but also records ``(provider,
    force_first)`` for every ``complete_grounded`` call, so a test can
    assert a specific tier's call actually forced ``tool_choice`` rather
    than just asking for a search in prose."""
    from types import SimpleNamespace

    from krepis.llm_config import LLMConfigError

    remaining = {k: list(v) for k, v in plan.items()}
    calls = []

    class _FakeClient:
        def __init__(self, spec, **kw):
            self.spec = spec

        def complete_grounded(self, *, search, **kw):
            calls.append((self.spec.provider, search.force_first))
            if self.spec.provider not in ("anthropic", "openrouter"):
                raise LLMConfigError(
                    f"complete_grounded unsupported on {self.spec.provider}"
                )
            queue = remaining.get(self.spec.provider)
            if not queue:
                raise AssertionError(
                    f"no more scripted responses for provider={self.spec.provider!r}"
                )
            return queue.pop(0)

        def complete(self, **kw):
            calls.append((self.spec.provider, None))
            queue = remaining.get(self.spec.provider)
            if not queue:
                raise AssertionError(
                    f"no more scripted complete() responses for "
                    f"provider={self.spec.provider!r}"
                )
            gr = queue.pop(0)
            if isinstance(gr, SimpleNamespace):
                return gr
            return SimpleNamespace(
                text=gr.text,
                model=gr.model,
                provider=gr.provider,
                usage=gr.usage,
                raw_request=gr.raw_request,
                raw_response=gr.raw_response,
            )

    return _FakeClient, calls


def _base_config(**overrides):
    cfg = {
        "llm": '{"provider": "openrouter", "model": "moonshotai/kimi-k2.6", "reasoning": {"exclude": true}}',
        "claude_model": "claude-haiku-4-5",
        "max_tokens": 4096,
        "web_search_max_uses": 20,
        "min_grounding_citations": 1,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def _patched(monkeypatch):
    monkeypatch.setattr(claude, "load_prompt", lambda weekend=False: "SYSTEM PROMPT")
    monkeypatch.setattr(claude, "load_news_context", lambda config, run_date=None: "")
    monkeypatch.setattr(claude, "is_non_trading_day", lambda date_str: False)
    monkeypatch.setattr(claude, "record_result_cost", lambda **kw: 0.0)
    monkeypatch.setattr(claude, "record_search_events",
                        lambda **kw: len(kw["searches"]))
    monkeypatch.setattr(claude, "capture_llm_call", lambda *a, **kw: False)
    monkeypatch.delenv(claude.LLM_ENV_VAR, raising=False)


def _decision_path(tmp_path, date_str="2026-07-06", edition="am"):
    return tmp_path / f"{date_str}-{edition}.llm_decision.json"


def test_falls_back_when_primary_produces_empty_content(monkeypatch, tmp_path):
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="", n_searches=10),
        ],
        "anthropic": [
            _grounded(provider="anthropic", model="claude-haiku-4-5",
                      text="Welcome to Morning Signal. Real content here.",
                      n_searches=5),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    script = claude.generate_script(_base_config(), "2026-07-06", "am")

    assert "Real content here" in script

    decision = json.loads(_decision_path(tmp_path).read_text())
    assert decision["primary_provider"] == "openrouter"
    assert decision["used_provider"] == "anthropic"
    assert decision["fell_back"] is True
    assert decision["primary_outcome"]["script_chars"] == 0
    assert decision["fallback_outcome"]["script_chars"] > 0


def test_falls_back_when_primary_has_no_search_results(monkeypatch, tmp_path):
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="some text but too few searches", n_searches=0,
                      citations=[]),
        ],
        "anthropic": [
            _grounded(provider="anthropic", model="claude-haiku-4-5",
                      text="Welcome to Morning Signal. Fallback content.",
                      n_searches=3),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    script = claude.generate_script(_base_config(), "2026-07-06", "am")

    assert "Fallback content" in script
    decision = json.loads(_decision_path(tmp_path).read_text())
    assert decision["fell_back"] is True
    assert decision["primary_outcome"] is None  # the guard raised before an outcome existed


def test_hard_aborts_when_both_primary_and_fallback_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="", n_searches=10),
        ],
        "anthropic": [
            _grounded(provider="anthropic", model="claude-haiku-4-5",
                      text="", n_searches=5),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    with pytest.raises(SystemExit):
        claude.generate_script(_base_config(), "2026-07-06", "am")

    # The decision log still records the (failed) attempt — worth knowing
    # both models failed today, not just silence.
    decision = json.loads(_decision_path(tmp_path).read_text())
    assert decision["fell_back"] is True
    assert decision["fallback_outcome"]["script_chars"] == 0


# ── Tier 3: forced-search Anthropic last-resort (morning-signal-I118,
# 2026-07-16) ────────────────────────────────────────────────────────────
#
# 2026-07-16 production incident: primary (Kimi) leaked a raw tool-call
# token, correctly triggered the configured (non-Anthropic) fallback
# (DeepSeek v4-flash) — which completed cleanly but never invoked
# web_search at all, tripping the zero-citations grounding guard with no
# further tier to catch it. These tests exercise the tier-3 rescue added
# to close that gap.


def test_tier3_forced_search_rescues_after_configured_fallback_also_fails(
    monkeypatch, tmp_path
):
    """Mirrors the 2026-07-16 failure exactly: primary produces no usable
    content, the CONFIGURED (non-Anthropic) fallback hard-fails its own
    grounding check (zero citations) — tier 3's forced-search Anthropic
    last-resort rescues the episode instead of aborting outright."""
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="", n_searches=10),
            _grounded(provider="openrouter", model="deepseek/deepseek-v4-flash",
                      text="ungrounded text", n_searches=0, citations=[]),
        ],
        "anthropic": [
            _grounded(provider="anthropic", model="claude-haiku-4-5",
                      text="Welcome to Morning Signal. Tier 3 content.",
                      n_searches=4),
        ],
    }
    FakeClient, calls = _client_factory_with_capture(plan)
    monkeypatch.setattr(claude, "LLMClient", FakeClient)

    config = _base_config(fallback_llm=(
        '{"provider": "openrouter", "model": "deepseek/deepseek-v4-flash", '
        '"reasoning": {"exclude": true}}'
    ))
    script = claude.generate_script(config, "2026-07-06", "am")

    assert "Tier 3 content" in script
    decision = json.loads(_decision_path(tmp_path).read_text())
    assert decision["used_provider"] == "anthropic"
    assert decision["used_model"] == "claude-haiku-4-5"
    assert decision["fell_back"] is True

    # The whole point of the fix: tier 3's call must have forced search,
    # not merely asked for one in prose (which is what tier 2 effectively
    # did and still produced zero citations).
    anthropic_calls = [force for provider, force in calls if provider == "anthropic"]
    assert anthropic_calls == [True]


def test_hard_aborts_when_forced_search_tier3_also_gets_zero_citations(
    monkeypatch, tmp_path
):
    """Even the forced-search last resort can fail its own grounding check
    (e.g. a genuine Anthropic outage) — must still hard-abort. There is
    nowhere further to fall back to."""
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="", n_searches=10),
            _grounded(provider="openrouter", model="deepseek/deepseek-v4-flash",
                      text="ungrounded text", n_searches=0, citations=[]),
        ],
        "anthropic": [
            _grounded(provider="anthropic", model="claude-haiku-4-5",
                      text="still ungrounded", n_searches=0, citations=[]),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    config = _base_config(fallback_llm=(
        '{"provider": "openrouter", "model": "deepseek/deepseek-v4-flash", '
        '"reasoning": {"exclude": true}}'
    ))

    with pytest.raises(RuntimeError, match="zero citations"):
        claude.generate_script(config, "2026-07-06", "am")

    # Exception propagates before the decision log would be written, same
    # as the anthropic-only-config hard-abort case.
    assert not _decision_path(tmp_path).exists()


def test_tier3_skipped_when_configured_fallback_is_already_anthropic(monkeypatch, tmp_path):
    """When no ``fallback_llm`` is configured, tier 2 already IS the
    Anthropic default — tier 3 must not fire a second, redundant Anthropic
    call. Unchanged pre-existing 2-tier behavior."""
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="", n_searches=10),
        ],
        "anthropic": [
            _grounded(provider="anthropic", model="claude-haiku-4-5",
                      text="", n_searches=5),
        ],
    }
    FakeClient, calls = _client_factory_with_capture(plan)
    monkeypatch.setattr(claude, "LLMClient", FakeClient)

    with pytest.raises(SystemExit):
        claude.generate_script(_base_config(), "2026-07-06", "am")

    # Exactly one anthropic call (tier 2), not two — no redundant tier 3.
    anthropic_calls = [p for p, _ in calls if p == "anthropic"]
    assert len(anthropic_calls) == 1


def test_no_fallback_when_primary_is_already_anthropic(monkeypatch, tmp_path):
    """Anthropic-only config: the content-grounding guard hard-aborts
    immediately on zero citations, no wasted second call to the same model."""
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    plan = {
        "anthropic": [
            _grounded(provider="anthropic", model="claude-haiku-4-5",
                      text="not enough searching", n_searches=0,
                      citations=[]),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    config = _base_config()
    config.pop("llm")  # legacy anthropic-default resolution path

    with pytest.raises(RuntimeError, match="zero citations"):
        claude.generate_script(config, "2026-07-06", "am")

    # No decision log at all — the exception propagates before we'd write one.
    assert not _decision_path(tmp_path).exists()


def test_no_fallback_when_env_override_pins_exact_spec(monkeypatch, tmp_path):
    """MORNING_SIGNAL_LLM is the operator/test escape hatch: it means run
    EXACTLY this spec, not "with a hidden fallback"."""
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    monkeypatch.setenv(claude.LLM_ENV_VAR, "openrouter:moonshotai/kimi-k2.6")
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="", n_searches=10),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    with pytest.raises(SystemExit):
        claude.generate_script(_base_config(), "2026-07-06", "am")


def test_decision_log_written_on_ordinary_success_no_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="Welcome to Morning Signal. All good.", n_searches=8),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    script = claude.generate_script(_base_config(), "2026-07-06", "am")

    assert "All good" in script
    decision = json.loads(_decision_path(tmp_path).read_text())
    assert decision["primary_provider"] == "openrouter"
    assert decision["used_provider"] == "openrouter"
    assert decision["fell_back"] is False


# ── S3 sync of the decision log (console visibility, 2026-07-06) ────────────


class _FakeS3:
    """No-op S3 client stand-in — records put_object calls, never touches
    the network (mirrors scripts/oss_bakeoff.py's test fake)."""

    def __init__(self):
        self.puts = []

    def put_object(self, **kw):
        self.puts.append(kw)


def test_decision_log_synced_to_s3_when_bucket_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    fake_s3 = _FakeS3()
    monkeypatch.setattr(claude, "_aws_client", lambda *a, **kw: fake_s3)
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="Welcome to Morning Signal. All good.", n_searches=8),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    claude.generate_script(_base_config(s3_bucket="test-bucket"), "2026-07-06", "am")

    assert len(fake_s3.puts) == 1
    put = fake_s3.puts[0]
    assert put["Bucket"] == "test-bucket"
    assert put["Key"] == "ops/llm_decisions/2026-07-06-am.llm_decision.json"
    synced = json.loads(put["Body"])
    assert synced["used_provider"] == "openrouter"


def test_decision_log_sync_skipped_without_s3_bucket_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    fake_s3 = _FakeS3()
    monkeypatch.setattr(claude, "_aws_client", lambda *a, **kw: fake_s3)
    config = _base_config()
    config.pop("s3_bucket", None)
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="Welcome to Morning Signal. All good.", n_searches=8),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    claude.generate_script(config, "2026-07-06", "am")

    assert fake_s3.puts == []
    # Local copy is unaffected regardless.
    assert _decision_path(tmp_path).exists()


def test_decision_log_sync_failure_does_not_block_publish(monkeypatch, tmp_path):
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    class _BrokenS3:
        def put_object(self, **kw):
            raise RuntimeError("simulated S3 outage")

    monkeypatch.setattr(claude, "_aws_client", lambda *a, **kw: _BrokenS3())
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="Welcome to Morning Signal. All good.", n_searches=8),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    script = claude.generate_script(_base_config(s3_bucket="test-bucket"), "2026-07-06", "am")

    assert "All good" in script
    assert _decision_path(tmp_path).exists()


# ── litellm / krepis router model-group primary ─────────────────────────────
#
# Production now points llm at the krepis router ``high`` group
# (provider=litellm, model=high). That transport has no server-side web
# search; grounding comes from the pre-fetched news_context digest. These
# tests lock the cascade behaviour for that path.


def test_router_group_resolves_to_the_authenticated_edge(monkeypatch, tmp_path):
    """A router-group `llm` spec is resolved through `krepis.router`, not
    handed to the in-process LiteLLM transport.

    `provider: litellm` in config means "the krepis `high` group"; what must
    reach `LLMClient` is the EDGE — `provider=litellm_proxy` with the edge's
    base_url and this consumer's credential name. Resolving it to the
    in-process Router instead is alpha-engine-config-I6367's forbidden
    linkage, and is what production silently did until 2026-08-08.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    edge_spec = ModelSpec(
        "litellm_proxy", "high",
        base_url="https://router.nousergon.ai:8443",
        api_key_env="ROUTER_CONSUMER_MORNINGSIGNAL",
        max_tokens=4096,
    )
    seen: dict = {}

    def _fake_resolve(group, *, exec_context=None, max_tokens=None, wire=None, **kw):
        seen["group"] = group
        seen["exec_context"] = exec_context
        seen["wire"] = wire
        return edge_spec, {"route": "litellm_proxy"}

    monkeypatch.setattr("krepis.router.resolve_group_spec", _fake_resolve)
    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ec2")

    plan = {
        "litellm_proxy": [
            _grounded(provider="litellm_proxy", model="deepseek-v4-pro",
                      text="Welcome to Morning Signal. Edge-served content.",
                      n_searches=0, citations=[]),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    script = claude.generate_script(
        _base_config(
            llm='{"provider": "litellm", "model": "high"}',
            min_grounding_citations=0,
        ),
        "2026-08-08", "am",
    )

    assert "Edge-served content" in script
    # `wire` is asserted because krepis' DEFAULT_WIRE is WIRE_ANTHROPIC —
    # inheriting it would let a substituted entry return a URL this
    # OpenAI-compatible transport cannot speak.
    assert seen == {"group": "high", "exec_context": "ec2", "wire": "openai"}
    decision = json.loads(_decision_path(tmp_path, "2026-08-08").read_text())
    assert decision["primary_provider"] == "litellm_proxy"
    assert decision["fell_back"] is False


def test_unresolvable_router_group_falls_back_and_alerts(monkeypatch, tmp_path):
    """Resolution failing is a DEPLOYMENT failure, not a provider one.

    R20 (`model-router-policy`) says a failed resolution fails closed rather
    than falling through to a default endpoint or an ambient key. Closed here
    means: do not call the group at all, run the configured fallback, and say
    so — the same treatment a primary that was never installed gets, because
    nothing about the next run will differ either.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    def _fake_resolve(group, **kw):
        raise RuntimeError("no reachable entry for group 'high'")

    monkeypatch.setattr("krepis.router.resolve_group_spec", _fake_resolve)

    class _AnthropicOnly:
        def __init__(self, spec, **kw):
            self.spec = spec
            assert spec.provider != "litellm", (
                "an unresolvable group must never be handed to the in-process "
                "LiteLLM transport"
            )

        def complete_grounded(self, **kw):
            return _grounded(
                provider="anthropic", model="claude-haiku-4-5",
                text="Welcome to Morning Signal. Aired on the fallback.",
                n_searches=4,
            )

    monkeypatch.setattr(claude, "LLMClient", _AnthropicOnly)

    sent: list[str] = []
    monkeypatch.setattr(
        "morning_signal.watchdog.send_alert",
        lambda config, edition, message: sent.append(message) or True,
    )

    script = claude.generate_script(
        _base_config(llm='{"provider": "litellm", "model": "high"}'),
        "2026-08-08", "am",
    )

    assert "Aired on the fallback" in script
    assert len(sent) == 1
    assert "never callable from this deployment" in sent[0]
    decision = json.loads(_decision_path(tmp_path, "2026-08-08").read_text())
    # The decision log records the DECLARED primary, not the fallback —
    # reporting the fallback there would erase the fact that the configured
    # primary was never reachable. The provider is normalised to `router`
    # (the legacy `litellm` spelling names an in-process transport krepis
    # refuses to construct); the group is preserved exactly, which is the
    # half that matters for diagnosis.
    assert decision["primary_provider"] == "router"
    assert decision["primary_model"] == "high"
    assert decision["used_provider"] == "anthropic"
    assert decision["fell_back"] is True


def test_group_resolving_to_a_direct_provider_is_refused(monkeypatch, tmp_path):
    """Resolving to a DIRECT provider is a failure, not a success.

    krepis skips the `litellm_proxy` route when its health probe fails or this
    consumer's credential does not resolve, and continues down the chain to a
    direct entry — for `high`, OpenRouter. Accepting that would reproduce the
    exact linkage alpha-engine-config-I6367 forbids, while looking like a
    working migration.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    direct_spec = ModelSpec(
        "openrouter", "deepseek/deepseek-v4-pro",
        api_key_env="OPENROUTER_API_KEY", max_tokens=4096,
    )
    monkeypatch.setattr(
        "krepis.router.resolve_group_spec",
        lambda group, **kw: (direct_spec, {"route": "direct"}),
    )

    class _AnthropicOnly:
        def __init__(self, spec, **kw):
            self.spec = spec
            assert spec.provider == "anthropic", (
                f"only the configured fallback may be called, got {spec.provider!r}"
            )

        def complete_grounded(self, **kw):
            return _grounded(
                provider="anthropic", model="claude-haiku-4-5",
                text="Welcome to Morning Signal. Aired on the fallback.",
                n_searches=4,
            )

    monkeypatch.setattr(claude, "LLMClient", _AnthropicOnly)

    sent: list[str] = []
    monkeypatch.setattr(
        "morning_signal.watchdog.send_alert",
        lambda config, edition, message: sent.append(message) or True,
    )

    script = claude.generate_script(
        _base_config(llm='{"provider": "litellm", "model": "high"}'),
        "2026-08-08", "am",
    )

    assert "Aired on the fallback" in script
    assert len(sent) == 1
    decision = json.loads(_decision_path(tmp_path, "2026-08-08").read_text())
    assert decision["used_provider"] == "anthropic"


def test_router_group_primary_succeeds_via_complete(monkeypatch, tmp_path):
    """A router-group primary produces a script through complete() — no
    web-search grounding check fires, no cascade needed.

    The edge speaks OpenAI-compatible chat completions and has no server-side
    web search, so grounding comes from the pre-fetched news_context digest.
    Was `test_litellm_primary_succeeds_via_complete`, asserting
    `used_provider == "litellm"` — the in-process Router. That assertion
    locked in the linkage alpha-engine-config-I6367 forbids; the degrade
    behaviour it was really testing is what survives here.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    edge_spec = ModelSpec(
        "litellm_proxy", "high",
        base_url="https://router.nousergon.ai:8443",
        api_key_env="ROUTER_CONSUMER_MORNINGSIGNAL",
        max_tokens=4096,
    )
    monkeypatch.setattr(
        "krepis.router.resolve_group_spec",
        lambda group, **kw: (edge_spec, {"route": "litellm_proxy"}),
    )

    plan = {
        "litellm_proxy": [
            _grounded(provider="litellm_proxy", model="deepseek-v4-pro",
                      text="Welcome to Morning Signal. High-group content.",
                      n_searches=0, citations=[]),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    script = claude.generate_script(
        _base_config(
            llm='{"provider": "litellm", "model": "high"}',
            # news-context-grounded path: disable the citation floor so an
            # empty-citations result is accepted (production does this by
            # skipping the guard entirely for non-web-search providers).
            min_grounding_citations=0,
        ),
        "2026-08-02", "am",
    )

    assert "High-group content" in script
    decision = json.loads(_decision_path(tmp_path, "2026-08-02").read_text())
    assert decision["primary_provider"] == "litellm_proxy"
    assert decision["used_provider"] == "litellm_proxy"
    assert decision["fell_back"] is False


def test_primary_unusable_in_this_deployment_alerts(monkeypatch, tmp_path):
    """A primary that was never CALLABLE here alerts, on top of falling back.

    Live 2026-08-08: the litellm primary raised ``ModuleNotFoundError: No
    module named 'litellm'`` on every run for six days after #135 merged.
    Each episode logged the abort and then published on the OpenRouter
    fallback, so the only failure surface — the episode not appearing — never
    fired. The episode still ships; the point is that somebody is told.

    That exact spec can no longer be constructed — a router group now resolves
    to the edge — so the missing-transport-package case is exercised here on
    an OpenRouter primary, which is the same category and still reachable.
    Its router-group sibling is
    ``test_unresolvable_router_group_falls_back_and_alerts``.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    class _PrimaryNotInstalled:
        def __init__(self, spec, **kw):
            self.spec = spec

        def complete_grounded(self, **kw):
            if self.spec.provider == "openrouter":
                raise ModuleNotFoundError("No module named 'openai'")
            return _grounded(
                provider="anthropic", model="claude-haiku-4-5",
                text="Welcome to Morning Signal. Aired on the fallback.",
                n_searches=4,
            )

        def complete(self, **kw):
            raise ModuleNotFoundError("No module named 'openai'")

    monkeypatch.setattr(claude, "LLMClient", _PrimaryNotInstalled)

    sent: list[str] = []
    monkeypatch.setattr(
        "morning_signal.watchdog.send_alert",
        lambda config, edition, message: sent.append(message) or True,
    )

    script = claude.generate_script(_base_config(), "2026-08-08", "am")

    assert "Aired on the fallback" in script
    assert len(sent) == 1, "a permanently-unusable primary must raise exactly one alert"
    assert "never callable from this deployment" in sent[0]
    assert "openrouter" in sent[0]


def test_ordinary_provider_failure_does_not_alert(monkeypatch, tmp_path):
    """The counterpart to the test above: a provider that WAS called and
    failed is what the fallback chain is for. Alerting on it would page for a
    mechanism working as designed, which is how an alert stops being read.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    class _APIStatusError(Exception):
        pass

    class _FailThenAnthropic:
        def __init__(self, spec, **kw):
            self.spec = spec

        def complete_grounded(self, **kw):
            if self.spec.provider == "openrouter":
                raise _APIStatusError("Error code: 402 - insufficient credits")
            return _grounded(
                provider="anthropic", model="claude-haiku-4-5",
                text="Welcome to Morning Signal. Rescued by cascade.",
                n_searches=4,
            )

    monkeypatch.setattr(claude, "LLMClient", _FailThenAnthropic)

    sent: list[str] = []
    monkeypatch.setattr(
        "morning_signal.watchdog.send_alert",
        lambda config, edition, message: sent.append(message) or True,
    )

    claude.generate_script(_base_config(), "2026-08-08", "am")

    assert sent == []


def test_non_runtime_error_from_primary_engages_fallback(monkeypatch, tmp_path):
    """2026-08-02 incident: OpenRouter HTTP 402 (APIStatusError, NOT a
    RuntimeError) killed the cascade before the Anthropic ultimate tier
    could engage. Cascade must catch Exception, not just RuntimeError.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    class _APIStatusError(Exception):
        """Stand-in for openai.APIStatusError — not a RuntimeError subclass."""

    class _FailThenAnthropic:
        def __init__(self, spec, **kw):
            self.spec = spec

        def complete_grounded(self, **kw):
            if self.spec.provider == "openrouter":
                raise _APIStatusError(
                    "Error code: 402 - insufficient credits"
                )
            return _grounded(
                provider="anthropic", model="claude-haiku-4-5",
                text="Welcome to Morning Signal. Rescued by cascade.",
                n_searches=4,
            )

        def complete(self, **kw):
            raise AssertionError("complete() should not be reached for openrouter/anthropic")

    monkeypatch.setattr(claude, "LLMClient", _FailThenAnthropic)

    script = claude.generate_script(_base_config(), "2026-08-02", "am")

    assert "Rescued by cascade" in script
    decision = json.loads(_decision_path(tmp_path, "2026-08-02").read_text())
    assert decision["fell_back"] is True
    assert decision["used_provider"] == "anthropic"


def test_a_400_from_the_router_primary_alerts(monkeypatch, tmp_path):
    """The four-day silence of 2026-08-09 .. 2026-08-12.

    `20-router.conf` declared this consumer's per-consumer router credential
    but not the URL of the edge that understands it, so krepis addressed the
    router process on loopback and the router — which has no database to
    resolve a virtual key against — answered every call:

        400 {"error":{"message":"No connected db.","type":"no_db_connection"}}

    Every scheduled run aborted its configured primary on that 400 and aired
    from a fallback. Nothing fired, because a 400 is neither an ``ImportError``
    nor a ``RouterGroupUnresolvable`` — the only two classes
    ``_is_deployment_class_failure`` recognised. The episode still ships; the
    point is that somebody is told.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    class _APIStatusError(Exception):
        """Stand-in for openai.APIStatusError."""

    class _RouterRejects:
        def __init__(self, spec, **kw):
            self.spec = spec

        def complete_grounded(self, **kw):
            if self.spec.provider == "openrouter":
                raise _APIStatusError(
                    "Error code: 400 - {'error': {'message': 'No connected db.', "
                    "'type': 'no_db_connection', 'code': '400'}}"
                )
            return _grounded(
                provider="anthropic", model="claude-haiku-4-5",
                text="Welcome to Morning Signal. Aired on the fallback.",
                n_searches=4,
            )

        def complete(self, **kw):
            raise _APIStatusError("Error code: 400 - {'error': {'message': 'No connected db.'}}")

    monkeypatch.setattr(claude, "LLMClient", _RouterRejects)

    sent: list[str] = []
    monkeypatch.setattr(
        "morning_signal.watchdog.send_alert",
        lambda config, edition, message: sent.append(message) or True,
    )

    script = claude.generate_script(_base_config(), "2026-08-09", "am")

    assert "Aired on the fallback" in script
    assert len(sent) == 1, (
        "a primary rejected 400 by the router must alert — the next run sends "
        "the same request and gets the same 400"
    )
    assert "never callable from this deployment" in sent[0]


def test_a_402_still_does_not_alert(monkeypatch, tmp_path):
    """The boundary the 400 case must not have moved.

    402 is a billing condition on the provider ACCOUNT, not a statement that
    this deployment is wired wrong, and `_is_deployment_class_failure`'s
    original contract names it explicitly as something the fallback chain
    exists for. Widening to "any 4xx" would have swept it in.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    class _APIStatusError(Exception):
        pass

    class _FailThenAnthropic:
        def __init__(self, spec, **kw):
            self.spec = spec

        def complete_grounded(self, **kw):
            if self.spec.provider == "openrouter":
                raise _APIStatusError("Error code: 402 - insufficient credits")
            return _grounded(
                provider="anthropic", model="claude-haiku-4-5",
                text="Welcome to Morning Signal. Rescued by cascade.",
                n_searches=4,
            )

    monkeypatch.setattr(claude, "LLMClient", _FailThenAnthropic)

    sent: list[str] = []
    monkeypatch.setattr(
        "morning_signal.watchdog.send_alert",
        lambda config, edition, message: sent.append(message) or True,
    )

    claude.generate_script(_base_config(), "2026-08-09", "am")

    assert sent == [], "a 402 is what the fallback chain is for"


def test_status_is_read_from_the_attribute_not_only_the_message():
    """Message parsing is a lossy SECOND source. An exception carrying the
    status as an attribute — which the OpenAI SDK does — must be classified
    without the string ever being read, or the classifier silently depends on
    a rendering the SDK is free to change.
    """
    class _WithAttr(Exception):
        status_code = 401

    class _WithAttrTransient(Exception):
        status_code = 429

    assert claude._is_client_error_status(_WithAttr("opaque, no status in text"))
    assert not claude._is_client_error_status(_WithAttrTransient("opaque"))


def test_the_registry_derived_egress_proxy_degrade_is_ACCEPTED(monkeypatch, tmp_path):
    """`model-router-policy` §5.2's degraded route is not a violation.

    When the edge's health probe fails, krepis walks the group's chain to the
    next registry entry `reachable_from` this context. MEASURED 2026-08-12 for
    group `high` / exec_context `ec2`, that is the **egress-proxy** DeepSeek
    entry — `deepseek-v4-pro-max`, `route=egress_proxy`, active — not a direct
    provider. The two OpenRouter rungs below it are marked `unavailable`.

    That route still traverses the DLP egress proxy, so R26 holds, and it was
    chosen by the registry rather than by this repo, which is exactly what §5.2
    requires. Refusing it (the prior behaviour: anything but `litellm_proxy`)
    made the one legitimate degraded path unreachable, so the consumer's own
    hardcoded cascade ran instead — direct OpenRouter, then direct Anthropic,
    on every episode from 2026-08-09 to 2026-08-12.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    degraded_spec = ModelSpec(
        "deepseek", "deepseek-v4-pro",
        base_url="http://127.0.0.1:8972", api_key_env="DEEPSEEK_API_KEY",
        max_tokens=4096,
    )
    monkeypatch.setattr(
        "krepis.router.resolve_group_spec",
        lambda group, **kw: (degraded_spec, {"route": "egress_proxy"}),
    )

    class _ServedOnTheDegradedRoute:
        def __init__(self, spec, **kw):
            self.spec = spec
            assert spec.base_url == "http://127.0.0.1:8972", (
                f"the degraded route must be the one the registry named, "
                f"got base_url={spec.base_url!r}"
            )

        def complete_grounded(self, **kw):
            return _grounded(
                provider="deepseek", model="deepseek-v4-pro",
                text="Welcome to Morning Signal. Aired on the compelled degraded route.",
                n_searches=4,
            )

    monkeypatch.setattr(claude, "LLMClient", _ServedOnTheDegradedRoute)

    sent: list[str] = []
    monkeypatch.setattr(
        "morning_signal.watchdog.send_alert",
        lambda config, edition, message: sent.append(message) or True,
    )

    cfg = _base_config(llm='{"provider": "litellm", "model": "high"}')
    script = claude.generate_script(cfg, "2026-08-12", "am")

    assert "compelled degraded route" in script, (
        "the episode must ship on the registry-derived degraded route rather "
        "than falling through to the consumer's own provider cascade"
    )
    # §5.4: every degraded-mode entry alerts, with the reason. Resolution
    # SUCCEEDED here, so without this nothing says the edge was not what served
    # it — a router unreachable for a week looks like one that is working.
    assert len(sent) == 1, "a degraded-route episode must raise exactly one alert"
    assert "DEGRADED" in sent[0]
    assert "egress_proxy" in sent[0]


def test_a_direct_provider_route_is_still_refused(monkeypatch, tmp_path):
    """The boundary the change above must not have moved.

    `egress_proxy` is compelled; `openrouter` is not. Widening to "any route
    krepis returns" would have re-admitted the alpha-engine-config-I6367
    linkage under a new name.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    direct_spec = ModelSpec(
        "openrouter", "deepseek/deepseek-v4-pro",
        api_key_env="OPENROUTER_API_KEY", max_tokens=4096,
    )
    monkeypatch.setattr(
        "krepis.router.resolve_group_spec",
        lambda group, **kw: (direct_spec, {"route": "openrouter"}),
    )

    with pytest.raises(claude.RouterGroupUnresolvable) as exc:
        claude._resolve_router_group(
            ModelSpec("router", "high", max_tokens=4096), _base_config()
        )
    assert "not a compelled path" in str(exc.value)
    assert "openrouter" in str(exc.value)


def test_the_degraded_route_alert_names_the_running_edition(monkeypatch, tmp_path):
    """`send_alert` passes the edition to `notify.make_doctor`, which uses it to
    pick the notification target — so a hardcoded value sends a PM-edition alert
    to the AM flow. The first version of `_alert_degraded_route` hardcoded
    "am"."""
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)

    degraded_spec = ModelSpec(
        "deepseek", "deepseek-v4-pro",
        base_url="http://127.0.0.1:8972", api_key_env="DEEPSEEK_API_KEY",
        max_tokens=4096,
    )
    monkeypatch.setattr(
        "krepis.router.resolve_group_spec",
        lambda group, **kw: (degraded_spec, {"route": "egress_proxy"}),
    )

    class _Served:
        def __init__(self, spec, **kw):
            self.spec = spec

        def complete_grounded(self, **kw):
            return _grounded(
                provider="deepseek", model="deepseek-v4-pro",
                text="Welcome to Morning Signal. Evening edition on the degraded route.",
                n_searches=4,
            )

    monkeypatch.setattr(claude, "LLMClient", _Served)

    editions: list[str] = []
    monkeypatch.setattr(
        "morning_signal.watchdog.send_alert",
        lambda config, edition, message: editions.append(edition) or True,
    )

    cfg = _base_config(llm='{"provider": "litellm", "model": "high"}')
    claude.generate_script(cfg, "2026-08-12", "pm")

    assert editions == ["pm"], (
        f"the degraded-route alert must name the running edition, got {editions!r}"
    )
