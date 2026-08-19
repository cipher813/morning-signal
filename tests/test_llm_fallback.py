"""Tests for the primary / fallback cascade (config#1659, retired to a
compelled-route-only shape by alpha-engine-config-I6980).

When the primary ``llm`` spec's own attempt (first pass + self-heal
recovery) either hard-fails (content-grounding verification) or silently
produces no usable content — a real, live-verified failure mode for
reasoning-capable models, 2026-07-06 — ``generate_script`` falls through to
ONE fresh attempt on the configured ``fallback_llm``. As of I6980,
``fallback_llm`` MUST name a krepis router group (same shape as ``llm``) and
is resolved through the identical ``_resolve_router_group`` path — it is no
longer a hardcoded provider slug, and there is no longer a silent
direct-Anthropic default when it is unset or a tier-3 forced-search
last-resort when it also fails. A compound failure of both configured tiers
now fails closed (`model-router-policy` §5.3) rather than getting a third
attempt: no episode, one alert (:func:`claude._alert_fallback_exhausted`).

Every call also writes a ``{date}-{edition}.llm_decision.json`` recording
which model actually produced (or failed to produce) the script.

Uses a duck-typed fake ``LLMClient`` (real krepis ``GroundedResult``/
``LLMUsage`` dataclasses, no real SDK/network) dispatched by provider, so
each test scripts exactly what the primary vs. fallback calls return. The
fallback's own ``krepis.router.resolve_group_spec`` call is mocked via
:func:`_mock_group_resolver`/:func:`_compelled_edge` the same way the
primary's already was.
"""

from __future__ import annotations

import json

import pytest
from krepis.llm import GroundedResult, LLMUsage
from krepis.llm_config import LLMConfigError, ModelSpec

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


#: A ``fallback_llm`` router-group config value — as of alpha-engine-config
#: I6980, `resolve_fallback_spec` requires this shape (same as `llm`) and
#: raises when it is unset or names a bare provider. Group name is
#: arbitrary; the resolution mock below is what decides what it resolves to.
_FALLBACK_GROUP = "med"
_FALLBACK_LLM_CONFIG = f'{{"provider": "router", "model": "{_FALLBACK_GROUP}"}}'


def _compelled_edge(provider, model, *, route="litellm_proxy", **spec_kw):
    """Build a ``(ModelSpec, route_dict)`` pair for a compelled route —
    what ``krepis.router.resolve_group_spec`` returns on success. Used to
    mock the fallback tier's own router-group resolution, same as the
    primary's (``_resolve_router_group`` treats both identically)."""
    spec_kw.setdefault("base_url", "https://router.nousergon.ai:8443")
    spec_kw.setdefault("api_key_env", "ROUTER_CONSUMER_MORNINGSIGNAL")
    spec_kw.setdefault("max_tokens", 4096)
    return ModelSpec(provider, model, **spec_kw), {"route": route}


def _mock_group_resolver(monkeypatch, groups):
    """Dispatch ``krepis.router.resolve_group_spec`` by group name.

    ``groups`` maps group name -> either a ``(spec, route_dict)`` tuple (see
    :func:`_compelled_edge`) or an exception instance to raise, so one test
    can script the primary group and the fallback group independently.
    """

    def _fake_resolve(group, **kw):
        entry = groups.get(group)
        if entry is None:
            raise AssertionError(f"unscripted group resolution request: {group!r}")
        if isinstance(entry, BaseException):
            raise entry
        spec, route = entry
        return spec, route

    monkeypatch.setattr("krepis.router.resolve_group_spec", _fake_resolve)


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
    _mock_group_resolver(monkeypatch, {
        _FALLBACK_GROUP: _compelled_edge("anthropic", "claude-haiku-4-5"),
    })
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

    script = claude.generate_script(
        _base_config(fallback_llm=_FALLBACK_LLM_CONFIG), "2026-07-06", "am"
    )

    assert "Real content here" in script

    decision = json.loads(_decision_path(tmp_path).read_text())
    assert decision["primary_provider"] == "openrouter"
    assert decision["used_provider"] == "anthropic"
    assert decision["fell_back"] is True
    assert decision["primary_outcome"]["script_chars"] == 0
    assert decision["fallback_outcome"]["script_chars"] > 0


def test_falls_back_when_primary_has_no_search_results(monkeypatch, tmp_path):
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    _mock_group_resolver(monkeypatch, {
        _FALLBACK_GROUP: _compelled_edge("anthropic", "claude-haiku-4-5"),
    })
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

    script = claude.generate_script(
        _base_config(fallback_llm=_FALLBACK_LLM_CONFIG), "2026-07-06", "am"
    )

    assert "Fallback content" in script
    decision = json.loads(_decision_path(tmp_path).read_text())
    assert decision["fell_back"] is True
    assert decision["primary_outcome"] is None  # the guard raised before an outcome existed


def test_hard_aborts_when_both_primary_and_fallback_fail(monkeypatch, tmp_path):
    """model-router-policy §5.3: a total compelled-path loss (here, both
    configured tiers producing no usable content) fails closed — one
    alert, no episode. Closes-when criterion #3 of alpha-engine-config
    I6980."""
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    _mock_group_resolver(monkeypatch, {
        _FALLBACK_GROUP: _compelled_edge("anthropic", "claude-haiku-4-5"),
    })
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

    sent: list[str] = []
    monkeypatch.setattr(
        "morning_signal.watchdog.send_alert",
        lambda config, edition, message: sent.append(message) or True,
    )

    with pytest.raises(SystemExit):
        claude.generate_script(
            _base_config(fallback_llm=_FALLBACK_LLM_CONFIG), "2026-07-06", "am"
        )

    # The decision log still records the (failed) attempt — worth knowing
    # both models failed today, not just silence.
    decision = json.loads(_decision_path(tmp_path).read_text())
    assert decision["fell_back"] is True
    assert decision["fallback_outcome"]["script_chars"] == 0

    assert len(sent) == 1, "a fully-exhausted cascade must fail closed with exactly one alert"
    assert "NO EPISODE" in sent[0]


# ── Fail-closed contract (alpha-engine-config-I6980) ─────────────────────
#
# The tier-3 forced-search Anthropic last-resort (morning-signal-I118,
# 2026-07-16) is RETIRED: its whole value was a deterministic ``tool_choice``
# force, an Anthropic-only transport capability, which coupled a
# grounding-quality guarantee to a direct-provider call — exactly the
# pattern this issue exists to end (direct-Anthropic is also a $0 budget
# line, 2026-07-17 ruling). These tests exercise what replaces it: a
# compound failure of both configured tiers now fails closed
# (model-router-policy §5.3) rather than getting a third attempt.


def test_fallback_llm_naming_a_bare_provider_is_refused(monkeypatch, tmp_path):
    """model-router-policy §5.2 permits only a registry-derived route; a
    bare provider slug in ``fallback_llm`` is refused OUTRIGHT — it must
    never reach an ``LLMClient`` call at all, matching the 2026-08-09..12
    incident where this exact shape served direct OpenRouter on every
    episode. Deliverable #1 of alpha-engine-config-I6980."""
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="", n_searches=10),
        ],
    }

    class _MustNeverBeCalledForFallback:
        def __init__(self, spec, **kw):
            self.spec = spec

        def complete_grounded(self, **kw):
            if self.spec.provider != "openrouter":
                raise AssertionError(
                    f"a bare-provider fallback_llm must never reach "
                    f"LLMClient, got provider={self.spec.provider!r}"
                )
            return plan["openrouter"].pop(0)

    monkeypatch.setattr(claude, "LLMClient", _MustNeverBeCalledForFallback)

    sent: list[str] = []
    monkeypatch.setattr(
        "morning_signal.watchdog.send_alert",
        lambda config, edition, message: sent.append(message) or True,
    )

    config = _base_config(fallback_llm=(
        '{"provider": "openrouter", "model": "deepseek/deepseek-v4-flash", '
        '"reasoning": {"exclude": true}}'
    ))

    # resolve_fallback_spec raises synchronously — no queued response is ever
    # consumed, so the exception is an LLMConfigError, not the "ran and
    # produced nothing" SystemExit path.
    with pytest.raises(LLMConfigError, match="bare provider"):
        claude.generate_script(config, "2026-07-06", "am")

    assert len(sent) == 1
    assert "NO EPISODE" in sent[0]


def test_hard_aborts_when_primary_and_fallback_both_fail_grounding(
    monkeypatch, tmp_path
):
    """Both configured tiers fail their own content-grounding check (zero
    citations) — must hard-abort. There is no tier 3 to catch this
    anymore; the RuntimeError from the fallback attempt propagates
    directly, and it is the ONE alert the fail-closed path fires."""
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    _mock_group_resolver(monkeypatch, {
        _FALLBACK_GROUP: _compelled_edge("anthropic", "claude-haiku-4-5"),
    })
    plan = {
        "openrouter": [
            _grounded(provider="openrouter", model="moonshotai/kimi-k2.6",
                      text="", n_searches=10),
        ],
        "anthropic": [
            _grounded(provider="anthropic", model="claude-haiku-4-5",
                      text="still ungrounded", n_searches=0, citations=[]),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    sent: list[str] = []
    monkeypatch.setattr(
        "morning_signal.watchdog.send_alert",
        lambda config, edition, message: sent.append(message) or True,
    )

    with pytest.raises(RuntimeError, match="zero citations"):
        claude.generate_script(
            _base_config(fallback_llm=_FALLBACK_LLM_CONFIG), "2026-07-06", "am"
        )

    # Exception propagates before the decision log would be written, same
    # as the anthropic-only-config hard-abort case — but the fail-closed
    # alert still fires (it is what tells a human the episode didn't ship).
    assert not _decision_path(tmp_path).exists()
    assert len(sent) == 1
    assert "NO EPISODE" in sent[0]


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

    _mock_group_resolver(monkeypatch, {
        "high": RuntimeError("no reachable entry for group 'high'"),
        _FALLBACK_GROUP: _compelled_edge("anthropic", "claude-haiku-4-5"),
    })

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
        _base_config(
            llm='{"provider": "litellm", "model": "high"}',
            fallback_llm=_FALLBACK_LLM_CONFIG,
        ),
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
    _mock_group_resolver(monkeypatch, {
        "high": (direct_spec, {"route": "direct"}),
        _FALLBACK_GROUP: _compelled_edge("anthropic", "claude-haiku-4-5"),
    })

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
        _base_config(
            llm='{"provider": "litellm", "model": "high"}',
            fallback_llm=_FALLBACK_LLM_CONFIG,
        ),
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


def test_tool_call_hallucination_on_no_search_transport_falls_back(monkeypatch, tmp_path):
    """Root cause, live 2026-08-19: every AM episode since the litellm_proxy
    migration (2026-08-13) aired 198-911 chars of unexecuted tool-call XML
    instead of a script. The prompt tells the model to call ``web_search``;
    complete() (the no-server-side-search degrade path) attaches no tools,
    so the model hallucinates one — and the content-grounding guards that
    would normally catch a hollow script are the exact guards this
    transport SKIPS (zero searches/citations is expected here by
    construction). The hallucination guard must fire regardless, and
    behave exactly like any other hard-abort of the primary attempt: fall
    through to the configured fallback rather than publish the garbage.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    litellm_spec = ModelSpec(
        "litellm_proxy", "high",
        base_url="https://router.nousergon.ai:8443",
        api_key_env="ROUTER_CONSUMER_MORNINGSIGNAL",
        max_tokens=4096,
    )
    fallback_spec, fallback_route = _compelled_edge("anthropic", "claude-haiku-4-5")
    monkeypatch.setattr(
        "krepis.router.resolve_group_spec",
        lambda group, **kw: (
            (fallback_spec, fallback_route) if group == _FALLBACK_GROUP
            else (litellm_spec, {"route": "litellm_proxy"})
        ),
    )
    plan = {
        "litellm_proxy": [
            _grounded(
                provider="litellm_proxy", model="deepseek-v4-pro",
                text=(
                    "Welcome to Morning Signal. <tool_calls>\n"
                    '<invoke name="search">\n'
                    '<parameter name="query" string="true">Trump Truth '
                    "Social posts August 19 2026</parameter>\n"
                    "</invoke>\n</tool_calls>"
                ),
                n_searches=0, citations=[],
            ),
        ],
        "anthropic": [
            _grounded(provider="anthropic", model="claude-haiku-4-5",
                      text="Welcome to Morning Signal. Real content here.",
                      n_searches=5),
        ],
    }
    monkeypatch.setattr(claude, "LLMClient", _client_factory(plan))

    script = claude.generate_script(
        _base_config(
            llm='{"provider": "litellm", "model": "high"}',
            fallback_llm=_FALLBACK_LLM_CONFIG,
            min_grounding_citations=0,
        ),
        "2026-08-19", "am",
    )

    assert "Real content here" in script
    assert "tool_calls" not in script
    decision = json.loads(_decision_path(tmp_path, "2026-08-19").read_text())
    assert decision["primary_provider"] == "litellm_proxy"
    assert decision["used_provider"] == "anthropic"
    assert decision["fell_back"] is True
    assert decision["primary_outcome"] is None  # the guard raised before an outcome existed


def test_complete_degrade_tells_the_model_it_has_no_tools(monkeypatch, tmp_path):
    """The root-cause fix: call_with_grounding_degrade's complete() path
    must override the prompt's "you will use web search" instruction —
    the call it's about to make has no tools attached. Asserts the actual
    payload sent to ``complete()``, not just the downstream behaviour.
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

    captured = {}

    class _CapturingClient:
        def __init__(self, spec, **kw):
            self.spec = spec

        def complete_grounded(self, **kw):
            raise LLMConfigError("complete_grounded unsupported on litellm_proxy")

        def complete(self, **kw):
            captured["user_content"] = kw["user_content"]
            from types import SimpleNamespace
            return SimpleNamespace(
                text="Welcome to Morning Signal. Real content.",
                model="deepseek-v4-pro", provider="litellm_proxy",
                usage=LLMUsage(web_search_requests=0),
                raw_request={}, raw_response=None,
            )

    monkeypatch.setattr(claude, "LLMClient", _CapturingClient)

    claude.generate_script(
        _base_config(
            llm='{"provider": "litellm", "model": "high"}',
            min_grounding_citations=0,
        ),
        "2026-08-19", "am",
    )

    assert "no web_search tool" in captured["user_content"].lower() or \
        "no tool" in captured["user_content"].lower()
    assert "ignore any instruction above to call" in captured["user_content"].lower()


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
    _mock_group_resolver(monkeypatch, {
        _FALLBACK_GROUP: _compelled_edge("anthropic", "claude-haiku-4-5"),
    })

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

    script = claude.generate_script(
        _base_config(fallback_llm=_FALLBACK_LLM_CONFIG), "2026-08-08", "am"
    )

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
    _mock_group_resolver(monkeypatch, {
        _FALLBACK_GROUP: _compelled_edge("anthropic", "claude-haiku-4-5"),
    })

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

    claude.generate_script(
        _base_config(fallback_llm=_FALLBACK_LLM_CONFIG), "2026-08-08", "am"
    )

    assert sent == []


def test_non_runtime_error_from_primary_engages_fallback(monkeypatch, tmp_path):
    """2026-08-02 incident: OpenRouter HTTP 402 (APIStatusError, NOT a
    RuntimeError) killed the cascade before the Anthropic ultimate tier
    could engage. Cascade must catch Exception, not just RuntimeError.
    """
    monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
    _mock_group_resolver(monkeypatch, {
        _FALLBACK_GROUP: _compelled_edge("anthropic", "claude-haiku-4-5"),
    })

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

    script = claude.generate_script(
        _base_config(fallback_llm=_FALLBACK_LLM_CONFIG), "2026-08-02", "am"
    )

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
    _mock_group_resolver(monkeypatch, {
        _FALLBACK_GROUP: _compelled_edge("anthropic", "claude-haiku-4-5"),
    })

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

    script = claude.generate_script(
        _base_config(fallback_llm=_FALLBACK_LLM_CONFIG), "2026-08-09", "am"
    )

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
    _mock_group_resolver(monkeypatch, {
        _FALLBACK_GROUP: _compelled_edge("anthropic", "claude-haiku-4-5"),
    })

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

    claude.generate_script(
        _base_config(fallback_llm=_FALLBACK_LLM_CONFIG), "2026-08-09", "am"
    )

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


class TestRouterPathDoesNotShadowTheRegistryBudget:
    """`max_tokens` is a registry-owned parameter (model-router-policy §2).

    Until fixed, `_resolve_router_group` passed
    `spec.max_tokens or config.get("max_tokens", 4096)` to
    `resolve_group_spec` — and `spec` is the inert declared ModelSpec, whose
    `max_tokens` krepis' dataclass forces to always be a truthy int. That
    `or` therefore NEVER short-circuited to `config.get(...)`, and
    `config.get(...)`'s own default meant this call NEVER once passed `None`
    to `resolve_group_spec` — so the registry's row for whichever model the
    group resolves to (`high` currently means DeepSeek V4 Pro Max, a
    reasoning model) was silently shadowed by this repo's own default on
    every single router call.

    Live-verified 2026-08-17 (alpha-engine-config, filed alongside this fix):
    identical to crucible-evaluator's Director bug
    (alpha-engine-config-I6396, crucible-evaluator#176) — "the call site's
    max_tokens=8000 shadowed the registry budget" — and to the reasoning-
    model budget-starvation class generally (alpha-engine-config-I6901):
    max_tokens bounds reasoning AND content from one shared pool, so a
    budget sized for a non-reasoning answer can return a fully-billed EMPTY
    response.
    """

    def test_unconfigured_max_tokens_passes_none_to_the_resolver(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
        edge_spec = ModelSpec(
            "litellm_proxy", "high",
            base_url="https://router.nousergon.ai:8443",
            api_key_env="ROUTER_CONSUMER_MORNINGSIGNAL",
            max_tokens=65536,
        )
        seen: dict = {}

        def _fake_resolve(group, *, max_tokens=None, **kw):
            seen["max_tokens"] = max_tokens
            return edge_spec, {"route": "litellm_proxy"}

        monkeypatch.setattr("krepis.router.resolve_group_spec", _fake_resolve)

        config = _base_config()
        del config["max_tokens"]
        claude._resolve_router_group(
            claude.declared_llm_spec(config), config
        )

        assert seen["max_tokens"] is None, (
            f"_resolve_router_group passed max_tokens={seen['max_tokens']!r} "
            "to resolve_group_spec with no operator override configured — "
            "this shadows the registry's row for the resolved model instead "
            "of deferring to it, resurrecting the Director/I6901 bug class."
        )

    def test_an_explicit_operator_override_still_reaches_the_resolver(
        self, monkeypatch, tmp_path
    ):
        """The override escape hatch (config.yaml.example's commented-out
        `max_tokens:`) must still work — this fix removes the SILENT
        default, not the ability to override deliberately."""
        monkeypatch.setattr(claude._config, "EPISODES_DIR", tmp_path)
        edge_spec = ModelSpec(
            "litellm_proxy", "high",
            base_url="https://router.nousergon.ai:8443",
            api_key_env="ROUTER_CONSUMER_MORNINGSIGNAL",
            max_tokens=65536,
        )
        seen: dict = {}

        def _fake_resolve(group, *, max_tokens=None, **kw):
            seen["max_tokens"] = max_tokens
            return edge_spec, {"route": "litellm_proxy"}

        monkeypatch.setattr("krepis.router.resolve_group_spec", _fake_resolve)

        config = _base_config(max_tokens=12000)
        claude._resolve_router_group(
            claude.declared_llm_spec(config), config
        )

        assert seen["max_tokens"] == 12000


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


# ── declared_llm_spec's unconfigured-`llm` default ──


def test_unset_llm_on_self_hosted_deployment_uses_the_legacy_anthropic_default(
    monkeypatch,
):
    """No `MORNING_SIGNAL_USE_SSM` (a self-hosted install, per README) with no
    `llm` configured MUST keep the pre-migration literal default: those
    installs have no krepis, no router, and no registry to derive a route
    from, so a hardcoded direct-Anthropic spec is correct there, not a
    model-router-policy violation (see `_anthropic_default_spec`'s
    docstring)."""
    monkeypatch.delenv("MORNING_SIGNAL_USE_SSM", raising=False)
    cfg = {"claude_model": "claude-haiku-4-5", "max_tokens": 111}

    spec = claude.declared_llm_spec(cfg)

    assert spec.provider == "anthropic"
    assert spec.model == "claude-haiku-4-5"
    assert spec.max_tokens == 111


def test_unset_llm_on_nous_ergon_deployment_fails_closed(monkeypatch):
    """`MORNING_SIGNAL_USE_SSM=1` (Nous Ergon's own deployment) with no `llm`
    configured is a deploy defect, not an absent user choice — production's
    SSM config-yaml always sets `llm`. model-router-policy §5.2/R20 forbids
    defaulting a compelled call site to a hardcoded direct-provider slug, so
    this MUST raise rather than silently reach for the OSS legacy default
    (which would spend on the $0-budget Anthropic route unannounced)."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    cfg = {"claude_model": "claude-haiku-4-5", "max_tokens": 111}

    with pytest.raises(claude.LLMConfigError, match="MORNING_SIGNAL_USE_SSM=1"):
        claude.declared_llm_spec(cfg)


def test_configured_llm_wins_over_the_ssm_fail_closed_check(monkeypatch):
    """A configured `llm` (the normal production case) must resolve
    normally on an SSM deployment — the fail-closed check above only fires
    when `llm` is actually unset."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    cfg = {"llm": '{"provider": "router", "model": "med"}', "max_tokens": 111}

    spec = claude.declared_llm_spec(cfg)

    assert spec.provider == "router"
    assert spec.model == "med"
