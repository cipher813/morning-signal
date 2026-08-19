"""Script generation via a web-search-grounded LLM call.

Runs through the krepis provider-agnostic adapter (``krepis.llm.LLMClient``).
Phase A of the fleet-wide provider migration (2026-07-03): generation stays on
the ANTHROPIC transport in production. The three production incident guards
below (content-grounding verification, ``required_search_topics`` coverage, and
forced-search recovery) are now provider-agnostic (config#1659 Phase B
re-key): they work off whichever signal a transport actually exposes —
Anthropic's per-query telemetry (``GroundedResult.searches``) when present,
falling back to citations (``GroundedResult.citations``) and the normalized
``usage.web_search_requests`` count on transports that don't expose queries
(OpenRouter's ``openrouter:web_search`` server tool). This is what lets the
``llm`` config knob (already the flip surface — see ``resolve_llm_spec``)
switch production between Anthropic, OpenAI, and OpenRouter models without
the safety net going dark. The LIVE flip itself stays gated on a ≥2-week
shadow-canary bakeoff (``scripts/oss_bakeoff.py``) per config#1659's
closes-when criteria — this module change is the prerequisite, not the flip.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

from krepis.llm import GroundedResult, LLMClient, SearchOptions
from krepis.llm_capture import capture_llm_call
from krepis.llm_config import LLMConfigError, ModelSpec, parse_model_spec
from krepis.trading_calendar import is_trading_day

from morning_signal import config as _config
from morning_signal.aws import _aws_client
from morning_signal.config import load_prompt
from morning_signal.cost_telemetry import record_result_cost
from morning_signal.news_context import load_news_context
from morning_signal.schedule_override import (
    derived_topic_guard,
    format_schedule_directive,
    load_schedule_override,
    record_applied,
)
from morning_signal.search_telemetry import (
    record_search_events,
    unmet_required_topics,
)

log = logging.getLogger("morning-signal")

# Transports that support server-side web search — anthropic (native
# web_search tool) and openrouter (openrouter:web_search server tool).
# Every other provider (litellm, openai, direct) lacks server-side search;
# grounding there comes from the pre-fetched news_context digest.
_PROVIDERS_WITH_WEB_SEARCH: frozenset[str] = frozenset({"anthropic", "openrouter"})

# Every prompt (prompt.md/prompt_weekend.md/prompt_public.md) unconditionally
# instructs the model to call a ``web_search`` tool ("you will use web search
# to gather fresh information", "≤5 web_search calls total"). That's true on
# the anthropic/openrouter transports but false on every other provider
# (litellm, openai, direct) — call_with_grounding_degrade's _call_complete()
# path issues a bare completion with NO tools attached. A model instructed to
# use a tool it was never given hallucinates one: it emits raw
# `<tool_calls>`/`<function_calls>`/`<invoke name="...">` XML as its entire
# response instead of script content. Nothing downstream caught this — the
# opener-check only prepends a missing opener, and the content-grounding
# guards are deliberately SKIPPED on this exact path (see
# `_news_context_grounded` in `_attempt_episode`) because it's expected to
# have zero searches/citations by construction. Root-caused live 2026-08-19:
# every AM episode since 2026-08-13 (the litellm_proxy router migration)
# aired 198-911 chars of tool-call XML instead of a script.
_TOOL_CALL_HALLUCINATION_RE = re.compile(
    r"<\s*(tool_calls|function_calls)\s*>|<\s*invoke\s+name\s*=", re.IGNORECASE
)


def verify_script_grounded(
    script: str,
    citations: list[dict] | None,
    *,
    date_str: str,
    edition_label: str,
    friendly_date: str,
    provider: str,
    min_citations: int = 1,
) -> None:
    """Verify the generated script is grounded in web search results.

    Replaces the raw search-count floor (``min_web_searches``) which was a
    proxy metric — counting tool invocations rather than measuring whether
    the model actually used search content at all. Different providers
    (Anthropic, OpenRouter/Kimi) search with very different cadences, so a
    count floor penalizes providers that do fewer, more targeted searches
    without telling you anything about output quality.

    This function checks the one thing that actually *is* provider-agnostic
    and meaningful: **did the search tool return substantive web content?**
    If it did, the model has the raw material to ground its output in live
    news. If it didn't (zero citations, all errors), the episode is almost
    certainly hallucinated — the same failure mode ``min_web_searches``
    existed to catch, but caught by the real signal instead of a proxy.

    Checks:
    1. **Citations exist** — at least one search result returned a citation
       with a real title. Cross-provider: Anthropic citations carry
       ``url`` + ``title``; OpenRouter citations carry ``url`` + ``title``
       + ``snippet``. A non-empty title proves the search tool worked and
       found a real page.
    2. **Date awareness** (WARN only) — the episode date appears in the
       script, indicating the model knows the current temporal context.
    3. **Anti-hallucination markers** (WARN only) — the script avoids
       training-data tells like "as of my last update" that indicate the
       model is relying on its parametric knowledge rather than search.

    Raises :exc:`RuntimeError` when check 1 fails: zero substantive
    citations means the search tool either wasn't called or returned no
    useful content — the episode is ungrounded and must not be published.
    """
    if not citations:
        raise RuntimeError(
            f"Search returned zero citations for {friendly_date} on "
            f"provider={provider!r} — the model likely did not call the "
            f"web_search tool at all. Refusing to publish an episode that "
            f"cannot be grounded in live web content. "
            f"(Set min_grounding_citations=0 in config to disable this check.)"
        )

    # Check 1: substantive citations (have a real title)
    substantive = [
        c for c in citations
        if c.get("title") and str(c.get("title") or "").strip()
    ]

    if len(substantive) < min_citations:
        raise RuntimeError(
            f"Only {len(substantive)}/{len(citations)} search citations "
            f"contained real web content (need ≥{min_citations}) for "
            f"{friendly_date} on provider={provider!r}. The search tool "
            f"returned URLs but the results had no substantive metadata — "
            f"likely the search hit errors or empty pages. Refusing to "
            f"publish an ungrounded episode."
        )

    log.info(
        f"Grounding check passed: {len(substantive)} substantive citations "
        f"from {len(citations)} total on {provider!r} — search returned "
        f"real web content."
    )

    # Check 2: date awareness (WARN only, never blocks)
    date_str_parts = date_str.replace("-", " ")
    script_lower = script.lower()
    if date_str not in script and date_str_parts not in script_lower:
        log.warning(
            f"Script for {friendly_date} does not mention the episode date "
            f"({date_str}) — the model may not have grounded the episode "
            f"in today's news. Publishing anyway since search citations "
            f"exist."
        )

    # Check 3: anti-hallucination markers (WARN only)
    hallucination_tells = [
        "as of my last update",
        "as of my knowledge cutoff",
        "i don't have access to real-time",
        "i don't have access to current",
        "according to my training data",
        "my training data only goes up to",
        "my knowledge cutoff",
    ]
    for tell in hallucination_tells:
        if tell in script_lower:
            log.warning(
                f"Possible hallucination marker detected in script for "
                f"{friendly_date}: contains {tell!r} — the model may be "
                f"relying on training data rather than search results. "
                f"Episode published but flag for review."
            )
            break

# Env override for the active ModelSpec (wins over config) — operator/test
# escape hatch. The durable flip surface is the ``llm`` key in config.yaml —
# which in production IS the /morning-signal/config-yaml SSM parameter, so a
# provider flip is an SSM edit picked up by the next cron firing, no redeploy.
LLM_ENV_VAR = "MORNING_SIGNAL_LLM"

# S3 mirror of the per-episode LLM decision log (see _record_llm_decision) —
# lets the console's Morning Signal Schedule page (crucible-dashboard) show
# which model aired each day without needing box access. Private: NOT under
# the public episodes/*|feed.xml|artwork.jpg prefixes the bucket policy
# allows anonymous reads on (2026-07-06 bucket-policy scoping) — readable
# only via the dashboard's own IAM grant on this exact prefix.
LLM_DECISION_S3_PREFIX = "ops/llm_decisions/"


def _anthropic_default_spec(config: dict) -> ModelSpec:
    """The anthropic-transport spec used as :func:`resolve_llm_spec`'s
    legacy default and as the final-layer fallback when no
    ``fallback_llm`` is configured.

    Not a router path — there is no krepis registry row to defer to for a
    bare direct-Anthropic call, so a literal default is correct here. See
    :func:`_resolve_router_group` for why the router path must NOT do this.
    """
    return ModelSpec(
        "anthropic",
        config.get("claude_model", "claude-sonnet-4-6"),
        max_tokens=config.get("max_tokens", 4096),
    )


def resolve_fallback_spec(config: dict) -> ModelSpec:
    """The fallback ModelSpec used when the primary spec fails.

    Reads config ``fallback_llm`` (same format as ``llm``) and resolves it
    through ``_resolve_router_group`` — the SAME path the primary spec goes
    through. `model-router-policy` §5.2 permits only a registry-derived
    route reachable from this execution context; it does not distinguish
    "primary" from "fallback" tiers, so neither may pick a provider by name.

    Previously this called ``parse_model_spec`` directly on a bare provider
    slug and returned it uncontested — that is exactly how the cascade
    served direct OpenRouter and, when that also failed, a silent
    Anthropic default on every episode 2026-08-09..12
    (alpha-engine-config-I6980). There is no direct-provider default left:
    a resolution failure here must propagate so the caller fails closed
    (§5.3), not land on a route nobody chose.

    :raises LLMConfigError: ``fallback_llm`` is unset, or names a bare
        provider rather than a router group.
    :raises RouterGroupUnresolvable: the named group did not resolve to a
        compelled route (:data:`_COMPELLED_ROUTES`) from this deployment.
    """
    configured = config.get("fallback_llm")
    if not configured:
        raise LLMConfigError(
            "config 'fallback_llm' is unset. model-router-policy §5.3 "
            "forbids defaulting an unconfigured fallback to a direct "
            "provider — set it to a krepis router group, e.g. "
            '{"provider": "router", "model": "med"} or the legacy spelling '
            '"litellm:med".'
        )
    group = _router_group_from_raw(str(configured))
    if group is None:
        raise LLMConfigError(
            f"config 'fallback_llm' = {configured!r} names a bare provider, "
            f"not a krepis router group. model-router-policy §5.2 permits "
            f"only a registry-derived route this repo does not choose by "
            f"name — set 'fallback_llm' to a router group (provider "
            f"'router' or 'litellm', e.g. {{\"provider\": \"router\", "
            f"\"model\": \"med\"}})."
        )
    # This ModelSpec.max_tokens is decorative — krepis.llm_config.ModelSpec
    # requires an int, but nothing reads this field back. The actual budget
    # is decided in _resolve_router_group, which reads config directly (not
    # this spec) so it can pass None through to the registry when unset. See
    # that function's comment for why.
    declared = ModelSpec("router", group, max_tokens=config.get("max_tokens", 4096))
    return _resolve_router_group(declared, config)


class RouterGroupUnresolvable(RuntimeError):
    """The configured spec names a krepis router group that could not be
    resolved into a callable endpoint from this deployment.

    A distinct type because the cascade must treat it as a *deployment*
    failure — permanent until something is changed here — rather than as a
    provider that was called and failed. See
    :func:`_is_deployment_class_failure`.
    """


#: Provider names in the ``llm`` config that mean "a krepis router group",
#: resolved through the authenticated router edge rather than called directly.
#:
#: ``router`` is the intended spelling. ``litellm`` is accepted because it is
#: what production's SSM config already says, and because a consumer naming it
#: is ALWAYS making the same mistake: in ``krepis.llm_config.PROVIDER_REGISTRY``
#: that name binds to ``TRANSPORT_LITELLM``, an in-process LiteLLM Router that
#: calls each provider directly from the consumer. krepis states the objection
#: in its own source (``router.py``, above ``resolve_group_spec``): that path
#: egresses straight to openrouter.ai unscanned — the linkage
#: alpha-engine-config-I6367 forbids — bypasses the authenticated edge so
#: per-consumer identity and rate limiting never apply, and requires the
#: ``litellm`` package plus a readable registry inside every consumer.
#:
#: Accepting the legacy name is what lets this migration land without an
#: operator editing ``/morning-signal/config-yaml`` first — a PR must be
#: deployable by the merge button alone.
_ROUTER_GROUP_PROVIDERS = frozenset({"router", "litellm"})

#: Routes this consumer may be served on — the COMPELLED PATHS.
#:
#: `litellm_proxy` is the authenticated router edge: the normal case, where the
#: proxy walks the group's chain server-side.
#:
#: `egress_proxy` is what `model-router-policy` §5.2 requires a consumer to
#: degrade to when the edge's health probe does not pass: a REGISTRY-DERIVED
#: direct route, declared `reachable_from` this execution context. It is still
#: the compelled path — the request traverses the DLP egress proxy, so R26
#: holds — and it is not a provider this repo chose.
#:
#: MEASURED 2026-08-12 for group `high`, exec_context `ec2`, which is what this
#: consumer resolves:
#:
#:   deepseek-v4-pro-max              route=egress_proxy  ec2  active
#:   grok-4                           route=egress_proxy  ec2  active
#:   qwen3-max                        route=openrouter         unavailable
#:   deepseek-v4-pro-openrouter-max   route=openrouter         unavailable
#:
#: So the degraded rung on this box is the egress-proxy DeepSeek entry, not a
#: direct provider. This check previously refused ANY route but `litellm_proxy`,
#: written on the assumption that the next rung was direct OpenRouter — true in
#: general, false here, and the cost of the assumption was that a healthy
#: compelled fallback was unreachable while the consumer's own hardcoded chain
#: (direct OpenRouter, then direct Anthropic) ran instead, on every episode from
#: 2026-08-09 to 2026-08-12.
#:
#: Anything NOT in this set — `openrouter`, `anthropic`, a bare provider — is
#: still refused: alpha-engine-config-I6367 forbids direct-OpenRouter linkage,
#: and the Anthropic API budget is $0 by the 2026-07-17 ruling.
_COMPELLED_ROUTES = frozenset({"litellm_proxy", "egress_proxy"})


def _resolve_router_group(spec: ModelSpec, config: dict) -> ModelSpec:
    """Turn a router-group spec into one addressing the authenticated edge.

    ``krepis.router.resolve_group_spec`` returns a ``ModelSpec`` carrying the
    edge's ``base_url`` and this consumer's ``api_key_env`` — `(url,
    credential)` and nothing else, which is the whole of what
    `model-router-policy` R27a says a consumer may need to know about the
    router. The model actually served, its fallback chain, and the per-provider
    endpoints are all decided above us and walked server-side by the proxy.

    Raises :exc:`RouterGroupUnresolvable` rather than degrading to anything.
    R20 is explicit that a failed resolution fails closed: falling through to a
    default endpoint or an ambient key spends real money on a route nobody
    chose. The caller engages the configured fallback and alerts.
    """
    try:
        from krepis.router import resolve_group_spec
    except ImportError as exc:  # pragma: no cover - krepis is a hard dependency
        raise RouterGroupUnresolvable(
            f"krepis.router is unavailable, so router group {spec.model!r} "
            f"cannot be resolved: {exc}"
        ) from exc

    # Declared, never inferred (R29). The value is supplied by the unit
    # (`KREPIS_EXEC_CONTEXT=ec2` on the dashboard box); passing None hands the
    # decision to krepis's own documented default rather than guessing here
    # from a hostname or a metadata lookup.
    exec_context = os.environ.get("KREPIS_EXEC_CONTEXT") or None
    try:
        edge_spec, route = resolve_group_spec(
            spec.model,
            exec_context=exec_context,
            # REQUIRED, not a default worth inheriting: krepis' DEFAULT_WIRE is
            # WIRE_ANTHROPIC. The router edge speaks OpenAI-compatible chat
            # completions, and every entry a fallback could substitute here is
            # reached the same way — so asking for the anthropic wire lets the
            # resolver hand back a URL this transport cannot speak, and the
            # failure surfaces as a malformed request rather than as a routing
            # decision. Mirrors alpha-engine-research's `single_agent.py`
            # challenger call site, which passes it explicitly for this reason.
            wire="openai",
            # config, NOT spec.max_tokens — `spec` is the inert declared
            # ModelSpec, whose max_tokens is ALWAYS a truthy int (krepis'
            # ModelSpec forces one; see declared_llm_spec/resolve_fallback_spec
            # above). Reading it here meant this call NEVER passed None, so
            # `resolve_group_spec` never once deferred to the registry's row
            # for whichever model the group resolves to — every router call
            # silently shadowed it with this repo's own default, the exact bug
            # crucible-evaluator's Director shipped and fixed
            # (alpha-engine-config-I6396, crucible-evaluator#176): "max_tokens
            # is a registry-owned parameter (model-router-policy §2); a literal
            # here wins over it with no warning and no trace in any log."
            #
            # `high` currently resolves to DeepSeek V4 Pro Max, a reasoning
            # model — max_tokens bounds reasoning AND content from ONE shared
            # pool (alpha-engine-config-I6901), so a config default sized for
            # a non-reasoning answer can starve content generation entirely.
            # config.get("max_tokens") is None unless an operator explicitly
            # set it, which is exactly when resolve_group_spec's contract says
            # to override the registry: "passing neither takes the registry
            # values, which is what a caller with no specific requirement
            # should do."
            max_tokens=config.get("max_tokens"),
        )
    except Exception as exc:
        raise RouterGroupUnresolvable(
            f"router group {spec.model!r} did not resolve "
            f"(exec_context={exec_context!r}): {exc}"
        ) from exc

    # Resolution SUCCEEDING is not the same as resolving to the router.
    #
    # krepis admits the `litellm_proxy` route only if its health probe passes
    # AND this consumer's credential resolves; when either fails the route is
    # skipped and resolution continues down the chain to a DIRECT provider
    # entry — for the `high` group, OpenRouter. That is a legitimate krepis
    # behaviour and a forbidden outcome here: alpha-engine-config-I6367 rules
    # out direct OpenRouter linkage, and the whole reason this repo addresses
    # a group by name is to stop choosing providers itself.
    #
    # So an unprovisioned credential must look like "unresolvable", not like
    # "resolved to something callable". Without this check the migration would
    # land looking successful and quietly reproduce the exact behaviour it was
    # written to end.
    resolved_route = route.get("route")
    if resolved_route not in _COMPELLED_ROUTES:
        raise RouterGroupUnresolvable(
            f"router group {spec.model!r} resolved to route "
            f"{resolved_route!r} (provider={edge_spec.provider!r}), which is "
            f"not a compelled path — refusing to call a direct provider chosen "
            f"by fallback (alpha-engine-config-I6367). Compelled routes: "
            f"{sorted(_COMPELLED_ROUTES)}"
        )

    if resolved_route != "litellm_proxy":
        # model-router-policy §5.4: every degraded-mode entry alerts, with the
        # reason. Resolution SUCCEEDED here — the episode will ship — so
        # nothing else on this path is going to say that the router edge was
        # not the thing that served it.
        log.warning(
            "DEGRADED: router group %r resolved to route %r (provider=%r "
            "model=%r) rather than the authenticated edge. The edge's health "
            "probe did not pass; this is the registry-derived degraded route "
            "(model-router-policy §5.2), which is still the compelled path.",
            spec.model, resolved_route, edge_spec.provider, edge_spec.model,
        )
        _alert_degraded_route(config, spec.model, resolved_route, edge_spec)

    log.info(
        "router group %r resolved to provider=%r model=%r route=%r",
        spec.model, edge_spec.provider, edge_spec.model, route.get("route"),
    )
    return edge_spec


def resolve_llm_spec(config: dict) -> ModelSpec:
    """The active ModelSpec: env override → config ``llm`` → legacy default.

    A spec naming a router group (see :data:`_ROUTER_GROUP_PROVIDERS`) is
    resolved through ``krepis.router`` into one addressing the authenticated
    router edge. The legacy default (anthropic + ``claude_model``) keeps every
    pre-migration config behavior-identical.

    :raises RouterGroupUnresolvable: the spec names a router group that could
        not be resolved. Deliberately not caught here — see
        :func:`_resolve_router_group`.
    """
    declared = declared_llm_spec(config)
    if declared.provider in _ROUTER_GROUP_PROVIDERS:
        return _resolve_router_group(declared, config)
    return declared


def _router_group_from_raw(raw: str) -> str | None:
    """The group name when *raw* is a router-group spec, else ``None``.

    Read WITHOUT going through ``parse_model_spec``, and that is the point:
    krepis refuses to construct a ``ModelSpec(provider="litellm")`` at all (it
    selects the in-process Router — see :data:`_ROUTER_GROUP_PROVIDERS`), so
    parsing production's live config value through it would raise before this
    module ever got to decide what the value MEANS. Reading the two fields
    here keeps the legacy spelling working against every krepis version,
    including the ones that refuse it.
    """
    text = raw.strip()
    provider = model = None
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except (TypeError, ValueError):
            return None
        if isinstance(obj, dict):
            provider, model = obj.get("provider"), obj.get("model")
    elif text.count(":") == 1:
        provider, _, model = text.partition(":")

    if provider in _ROUTER_GROUP_PROVIDERS and model:
        return str(model)
    return None


def declared_llm_spec(config: dict) -> ModelSpec:
    """The ``llm`` spec exactly as configured, with no router resolution.

    What the decision log should record as the PRIMARY even when resolution
    failed: `{"provider": "litellm", "model": "high"}` is what the operator
    asked for, and reporting the fallback in its place would erase the fact
    that the configured primary was never reachable.

    A router group is normalised to ``provider="router"`` — a name krepis does
    not know, so it constructs as an inert custom spec rather than selecting a
    transport. Nothing calls it; it exists to be RECORDED and to be recognised
    by :func:`resolve_llm_spec`.
    """
    raw = os.environ.get(LLM_ENV_VAR)
    source = f"env {LLM_ENV_VAR}"
    if not raw:
        configured = config.get("llm")
        if not configured:
            return _anthropic_default_spec(config)
        raw, source = str(configured), "config 'llm'"

    group = _router_group_from_raw(raw)
    if group is not None:
        # Decorative — see resolve_fallback_spec's comment on the identical
        # construction above. _resolve_router_group reads config directly.
        return ModelSpec(
            "router", group, max_tokens=config.get("max_tokens", 4096)
        )
    return parse_model_spec(raw, source=source)

EDITION_LABELS = {"am": "MORNING", "pm": "EVENING"}

# Sentinel distinguishing "caller didn't pass a schedule entry — load it
# yourself" from an explicit None ("no schedule entry applies", e.g. a
# --force run over a skip date). Keeps generate_script's original
# 3-positional-arg call contract intact for direct callers/tests while
# letting episode.main share its single manifest read.
_LOAD_SCHEDULE = object()


def _alert_degraded_coverage(
    config: dict,
    edition: str,
    edition_label: str,
    date_str: str,
    unmet: list[str],
    n_searches: int,
    budget: int,
) -> None:
    """Fire a flow-doctor/Telegram alert for an episode that shipped with one
    or more required search topics uncovered (see the degraded-coverage policy
    in :func:`generate_script`).

    Best-effort by construction: ``watchdog.send_alert`` already no-ops when
    notifications are disabled / creds are missing, and we additionally guard
    the whole call so an alerting bug can NEVER block the publish path. A
    send failure is logged at WARNING (so it is still a recorded surface, not
    a silent swallow) — the episode ships either way.
    """
    topics = "\n".join(f"  • {t}" for t in unmet)
    message = (
        f"⚠️ morning-signal {edition_label} edition for {date_str} "
        f"PUBLISHED WITH DEGRADED COVERAGE.\n\n"
        f"Required search topic(s) NOT covered by live web search "
        f"(likely written from memory rather than today's news):\n"
        f"{topics}\n\n"
        f"Ran {n_searches} web search(es); budget web_search_max_uses={budget}.\n"
        f"Triage: raise web_search_max_uses, revisit the segment prompt, or "
        f"adjust the required_search_topics keyword matchers. The episode "
        f"still shipped — fix forward."
    )
    try:
        from morning_signal.watchdog import send_alert

        send_alert(config, edition, message)
    except Exception:  # noqa: BLE001 — alerting must never block publish
        log.warning(
            "DEGRADED COVERAGE alert failed to send (continuing to publish)",
            exc_info=True,
        )


#: 4xx statuses that mean THIS DEPLOYMENT'S REQUEST could never be served —
#: its credential, its endpoint, or the model name it addressed. An allowlist
#: rather than "4xx minus the transient ones", because the 4xx range also
#: carries conditions that are about the provider ACCOUNT or its capacity
#: rather than about how this deployment is wired:
#:
#:   402  insufficient credits — a billing condition; the fallback chain
#:        exists for it, and this function's original contract names it
#:        explicitly as something not to fire on
#:   408  a timeout wearing a client-error number
#:   429  the rate limiter doing its job; says nothing about whether the
#:        request was well-formed
#:
#: Those stay absorbed. The ones below cannot come right on their own, because
#: the next scheduled run sends exactly the same request.
_DEPLOYMENT_CLASS_STATUSES = frozenset({400, 401, 403, 404, 405, 422})

#: The OpenAI SDK's rendering of an upstream status, e.g.
#: ``Error code: 400 - {'error': {...}}``. Parsed only as a FALLBACK for
#: exceptions that were re-raised as a plain message and lost the attribute;
#: the attribute below is the primary source.
_STATUS_IN_MESSAGE = re.compile(r"\bError code:\s*(\d{3})\b")


def _is_client_error_status(exc: BaseException) -> bool:
    """True when *exc* carries a status meaning this deployment's request could
    never be served — see :data:`_DEPLOYMENT_CLASS_STATUSES`.
    """
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        # Some layers re-raise with the status only in the message. Reading it
        # back is lossy, so it is a second source and never the only one.
        m = _STATUS_IN_MESSAGE.search(str(exc))
        status = int(m.group(1)) if m else None
    return status in _DEPLOYMENT_CLASS_STATUSES


#: Key under which the running edition is stamped into the config dict so that
#: spec resolution — which happens below the level that knows the edition — can
#: label its alerts correctly. Leading underscore because it is an in-process
#: annotation, never a configured value; nothing reads it from SSM or YAML.
_EDITION_KEY = "_edition"


def _alerting_edition(config: dict) -> str:
    """The edition an alert raised during spec resolution belongs to.

    `send_alert` passes this to `notify.make_doctor`, which uses it to pick the
    notification target — so a wrong value sends a PM-edition alert to the AM
    flow. The first version of `_alert_degraded_route` hardcoded ``"am"``,
    which is the same defect this session spent its length fixing elsewhere: a
    label that does not track the thing it names.

    Falls back to ``"am"`` only when the caller genuinely has no edition —
    `scripts/oss_bakeoff.py` resolves a spec outside any episode. That path has
    no fallback chain of its own, so it fails loudly rather than relying on
    this alert.
    """
    return str(config.get(_EDITION_KEY) or "am")


def _alert_degraded_route(
    config: dict, group: str, route: str, spec: ModelSpec
) -> None:
    """Fire an alert for an episode served off the authenticated router edge.

    ``model-router-policy`` §5.4: *every degraded-mode entry alerts, with the
    reason.* Resolution SUCCEEDED on this path — the episode ships — so without
    this nothing says the edge was not what served it, and a router that has
    been unreachable for a week looks exactly like one that is working.

    Best-effort by construction, same contract as
    :func:`_alert_degraded_coverage`: ``send_alert`` already no-ops when
    notifications are off, and the whole call is guarded so an alerting bug can
    never block the publish path. A send failure is logged at WARNING — a
    recorded surface, not a silent swallow.
    """
    message = (
        f"⚠️ morning-signal is running DEGRADED: router group {group!r} was "
        f"served on route {route!r}, not the authenticated router edge.\n\n"
        f"The edge's health probe did not pass, so krepis resolved the "
        f"registry-derived degraded route (model-router-policy §5.2). That is "
        f"still the compelled path — provider={spec.provider!r} "
        f"model={spec.model!r} — so nothing has left DLP scanning, and the "
        f"episode ships.\n\n"
        f"Triage: check litellm-proxy.service and the nginx router edge on the "
        f"dashboard box. This is not a per-episode fault; it will keep firing "
        f"until the edge is healthy."
    )
    try:
        from morning_signal.watchdog import send_alert

        send_alert(config, _alerting_edition(config), message)
    except Exception:  # noqa: BLE001 — alerting must never block publish
        log.warning(
            "DEGRADED ROUTE alert failed to send (continuing to publish)",
            exc_info=True,
        )


def _is_deployment_class_failure(exc: BaseException) -> bool:
    """True when *exc* means the configured primary spec was never CALLABLE
    from this deployment, as opposed to having been called and failed.

    The distinction is the whole point of alerting on one and not the other.
    A provider timeout, a 402, a content filter — those are what the fallback
    chain exists for, and firing on them would page daily for a mechanism
    working as designed. A missing transport package (``No module named
    'litellm'``) is not a provider event at all: the primary cannot succeed
    on the next run, or any run, until the deployment changes. Absorbing it
    into the fallback makes a permanent misconfiguration look like a
    transient one, and nothing else on the episode path distinguishes them.

    Found live 2026-08-08: morning-signal ran every episode on its OpenRouter
    fallback for six days after the primary was flipped to a krepis router
    group (#135, merged 2026-08-02) whose transport package was never an
    installed dependency. Every run logged the abort at ERROR and then
    published successfully, so no failure surface ever fired — and the
    fallback in question is direct-OpenRouter linkage, which
    alpha-engine-config-I6367 forbids. The silence, not the fallback, is
    what let it stand for six days.

    RECURRED 2026-08-09 .. 2026-08-12 with a different exception type, which is
    why the membership test below is no longer a list of two classes. The unit
    declared this consumer's per-consumer router credential but not the URL of
    the edge that understands it, so krepis addressed the router process on
    loopback and the router — which has no database to resolve a virtual key
    against — answered every call ``400 no_db_connection``. Four days, every
    scheduled run, aired from a fallback, silent: a 400 is neither an
    ``ImportError`` nor a ``RouterGroupUnresolvable``.

    A 4xx from the primary is a statement about the REQUEST — its credential,
    its endpoint, the model name it addressed — and the next run sends exactly
    the same request. It cannot come right on its own, so absorbing it into
    the fallback is the same mistake in a new costume. 5xx, timeouts and
    transport errors stay absorbed: those are what a fallback chain is FOR,
    and firing on them would page for a mechanism working as designed.
    ``LLMConfigError`` joins for the same reason — krepis raises it when the
    resolved spec and the router's answer are mutually inconsistent, which no
    retry changes.
    """
    seen: set[int] = set()
    cursor: BaseException | None = exc
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        # ImportError covers ModuleNotFoundError; RouterGroupUnresolvable means
        # the configured group never produced a callable endpoint here, which
        # is the same category — nothing about the next run will differ.
        if isinstance(cursor, (ImportError, RouterGroupUnresolvable, LLMConfigError)):
            return True
        if _is_client_error_status(cursor):
            return True
        cursor = cursor.__cause__ or cursor.__context__
    return False


def _alert_primary_never_usable(
    config: dict,
    edition: str,
    primary_spec: ModelSpec,
    exc: BaseException,
    *,
    friendly_date: str,
    edition_label: str,
) -> None:
    """Alert that the CONFIGURED primary spec is unusable in this deployment.

    Same best-effort posture as :func:`_alert_degraded_coverage`: the episode
    ships on the fallback either way, and an alerting bug must never be what
    takes the podcast off the air. A send failure is logged at WARNING rather
    than swallowed.
    """
    message = (
        f"🚨 morning-signal {edition_label} edition for {friendly_date}: "
        f"the CONFIGURED PRIMARY was never callable from this deployment.\n\n"
        f"Spec: provider={primary_spec.provider!r} model={primary_spec.model!r}\n"
        f"Error: {exc}\n\n"
        f"This is a deployment defect, not a provider failure — it will "
        f"recur on every run until the deployment changes, and every episode "
        f"until then airs on the FALLBACK spec. The episode still shipped."
    )
    try:
        from morning_signal.watchdog import send_alert

        send_alert(config, edition, message)
    except Exception:  # noqa: BLE001 — alerting must never block publish
        log.warning(
            "PRIMARY-UNUSABLE alert failed to send (continuing to publish)",
            exc_info=True,
        )


def _alert_fallback_exhausted(
    config: dict,
    edition: str,
    edition_label: str,
    friendly_date: str,
    primary_spec: ModelSpec,
    exc: BaseException | None,
) -> None:
    """Alert that NO compelled route produced a script — no episode today.

    `model-router-policy` §5.3: when no registry-derived route both resolves
    and serves the caller's wire format, the consumer fails closed. Not
    publishing is the CORRECT outcome here — publishing on a route outside
    :data:`_COMPELLED_ROUTES` is the forbidden alternative this issue exists
    to end. But a fail-closed episode with no alert is indistinguishable
    from a cron job that silently stopped firing, so this is what tells a
    human the difference between "the podcast is down" and "nothing ran".

    Unlike :func:`_alert_primary_never_usable` and
    :func:`_alert_degraded_route`, the episode does NOT ship after this — it
    fires exactly once, from the branch that is about to raise. Same
    best-effort posture as the other alerts: send failure is logged at
    WARNING, never swallowed, and never allowed to mask the real exception.
    """
    message = (
        f"🚨 morning-signal {edition_label} edition for {friendly_date}: "
        f"NO EPISODE — every compelled route was exhausted.\n\n"
        f"Configured primary: provider={primary_spec.provider!r} "
        f"model={primary_spec.model!r}\n"
        f"Last error: {exc}\n\n"
        f"model-router-policy §5.3: failing closed here is correct — the "
        f"forbidden alternative was publishing on a direct provider nobody "
        f"chose. This means the authenticated router edge AND its "
        f"registry-derived degraded route (§5.2) are both unavailable from "
        f"this deployment. Triage: litellm-proxy.service and the egress "
        f"proxy on the dashboard box; this will keep firing, with no "
        f"episode, until one of them is healthy again."
    )
    try:
        from morning_signal.watchdog import send_alert

        send_alert(config, edition, message)
    except Exception:  # noqa: BLE001 — alerting must never mask the raise
        log.warning(
            "FAIL-CLOSED alert failed to send (no episode either way)",
            exc_info=True,
        )


def is_non_trading_day(date_str: str) -> bool:
    """True for Sat/Sun + NYSE holidays. Drives prompt + PM-skip selection."""
    return not is_trading_day(datetime.strptime(date_str, "%Y-%m-%d").date())


def opening_line(edition: str, weekend: bool) -> str:
    """Canonical opening line every episode must begin with.

    Instructed via the dynamic user message and enforced via response
    post-processing. Previously sent as an assistant-turn prefill, but
    that combination with the ``web_search`` server tool is rejected
    by Anthropic — the producer-side guard lives in
    ``krepis.anthropic_payload.validate_payload``, which the krepis
    ``LLMClient`` anthropic transport runs at payload-construction time,
    so any future regression that re-introduces the prefill+server-tool
    combo fails loud before reaching the API.
    """
    if weekend:
        return "Welcome to Morning Signal, weekend edition."
    if edition == "pm":
        return "Welcome to Morning Signal, evening edition."
    return "Welcome to Morning Signal."


def call_with_grounding_degrade(
    llm_client: LLMClient,
    config: dict,
    prompt_text: str,
    user_content: str,
    *,
    force_search: bool = False,
) -> GroundedResult:
    """Issue one generation call, degrading to ``complete()`` on a transport
    with no server-side web search. Returns the ``GroundedResult``; records
    nothing.

    This is the single definition of *how morning-signal calls a model*, and
    it is deliberately separate from :func:`_invoke_and_record`, which is that
    call PLUS the aired episode's telemetry sinks. The split exists because
    ``scripts/oss_bakeoff.py`` needs the first and must not have the second:
    a shadow-canary comparison writing into ``episodes/{date}-{edition}.
    cost.jsonl`` would mix never-published measurement calls into the aired
    episode's billed-cost record.

    Before this was factored out, the bakeoff had its own copy of the call
    that was missing the degrade branch entirely — a bare
    ``complete_grounded``. That divergence was invisible while the production
    ``llm`` spec was an Anthropic or OpenRouter one (both have server-side
    search) and became a hard failure the moment the spec moved to a krepis
    router group (#135, merged 2026-08-02): production degraded correctly and
    kept airing, while the weekly bakeoff exited 1 from 2026-08-05 onward on
    ``LLMConfigError``. Two call sites, one of which had the fix, is the bug
    — so callers get the behaviour and own their own recording.
    """

    def _call_grounded(*, force: bool) -> GroundedResult:
        return llm_client.complete_grounded(
            system=prompt_text,
            user_content=user_content,
            search=SearchOptions(
                max_uses=config.get("web_search_max_uses", 20),
                force_first=force,
            ),
            # None unless an operator explicitly set config 'max_tokens' — see
            # _resolve_router_group's comment (alpha-engine-config-I6396,
            # I6901). A literal default here shadows the router-resolved
            # model's registry-owned budget on every call, reasoning models
            # included.
            max_tokens=config.get("max_tokens"),
            cache_system=True,
        )

    def _call_complete() -> GroundedResult:
        # The system prompt still tells the model "you will use web search"
        # (it's shared verbatim with the grounded transports — see the
        # _TOOL_CALL_HALLUCINATION_RE comment above). This call attaches no
        # tools, so override that instruction in the user turn — the one
        # part of the payload this path already varies per call — rather
        # than forking a second prompt file per transport.
        degraded_user_content = (
            f"{user_content}\n\n"
            f"IMPORTANT: this call has NO web_search tool or any other tool "
            f"attached — ignore any instruction above to call one. Write the "
            f"script using ONLY the news context already provided. Do not "
            f"emit a tool call, function call, or any XML/JSON invoking a "
            f"tool by name; there is nothing on the other end to receive it "
            f"and doing so will corrupt the aired episode."
        )
        complete_result = llm_client.complete(
            system=prompt_text,
            user_content=degraded_user_content,
            max_tokens=config.get("max_tokens"),
            cache_system=True,
            on_unsupported="drop",
        )
        return GroundedResult(
            text=complete_result.text,
            model=complete_result.model,
            provider=complete_result.provider,
            usage=complete_result.usage,
            raw_request=complete_result.raw_request,
            raw_response=complete_result.raw_response,
            searches=[],
            citations=[],
        )

    try:
        return _call_grounded(force=force_search)
    except LLMConfigError:
        if force_search:
            if llm_client.spec.provider not in _PROVIDERS_WITH_WEB_SEARCH:
                log.info(
                    f"provider={llm_client.spec.provider!r} does not support "
                    f"server-side web search — using news-context-grounded "
                    f"complete() instead. force_search is a no-op on this "
                    f"transport."
                )
                return _call_complete()
            log.warning(
                f"provider={llm_client.spec.provider!r} cannot force "
                f"web_search via tool_choice (SearchOptions.force_first "
                f"unsupported on this transport) — retrying the recovery "
                f"pass with prose-only forcing. Coverage on this pass is "
                f"best-effort, NOT hard-guaranteed the way it is on the "
                f"anthropic transport."
            )
            return _call_grounded(force=False)
        if llm_client.spec.provider not in _PROVIDERS_WITH_WEB_SEARCH:
            log.info(
                f"provider={llm_client.spec.provider!r} does not support "
                f"server-side web search — using news-context-grounded "
                f"complete() instead."
            )
            return _call_complete()
        raise


def _invoke_and_record(
    llm_client: LLMClient,
    config: dict,
    prompt_text: str,
    user_content: str,
    date_str: str,
    edition: str,
    force_search: bool = False,
):
    """Run one grounded generation pass and record its telemetry.

    Returns ``(result, n_searches)`` where ``result`` is a krepis
    ``GroundedResult``. Factored out of :func:`generate_script` so the
    self-healing recovery pass can re-invoke the model with an escalated user
    message under the SAME payload contract. Cost + per-search telemetry
    append to the episode's JSONL sinks on every call, so a recovery pass is
    captured as an additional billed call (accurate cost record), not hidden.

    ``force_search=True`` deterministically forces a ``web_search`` tool call
    (``SearchOptions.force_first`` → forced server-tool ``tool_choice`` on
    the anthropic transport) — used by the recovery pass rather than merely
    asking for a search in prose. Verified against the live API (2026-06-29).
    Some transports cannot force their server-side search tool at all — the
    OpenRouter ``openrouter:web_search`` tool has no ``tool_choice`` forcing,
    and ``krepis.llm`` raises ``LLMConfigError`` rather than silently
    degrading (see ``SearchOptions.force_first``). On that error this
    function retries the SAME call with ``force_first=False``: the recovery
    pass's escalated user message (``_coverage_recovery_directive``) already
    carries a strongly-worded prose forcing directive, so the retry still
    asks for the search — it just can't be hard-guaranteed on this
    transport. That degraded guarantee is logged explicitly, never silently
    treated as equivalent to a hard force.

    The krepis anthropic transport builds its payload through
    ``krepis.anthropic_payload`` — the server-tool ⊥ assistant-prefill
    invariant is still enforced at lib level. Every call is also staged to
    the distillation corpus (``episodes/{date}-{edition}.sft.jsonl``) when
    ``LLM_SFT_CAPTURE_ENABLED=1`` — a no-op otherwise, and a persist failure
    with the flag on raises rather than silently dropping training data
    (same disk the episode itself writes to).
    """

    result = call_with_grounding_degrade(
        llm_client, config, prompt_text, user_content,
        force_search=force_search,
    )

    cost = record_result_cost(
        result=result,
        date_str=date_str,
        edition=edition,
        episodes_dir=_config.EPISODES_DIR,
    )
    n_recorded = record_search_events(
        searches=result.searches,
        date_str=date_str,
        edition=edition,
        episodes_dir=_config.EPISODES_DIR,
    )
    # Provider-agnostic search count for the min_web_searches floor.
    # Anthropic populates BOTH result.searches (per-query, what
    # record_search_events counts) and usage.web_search_requests (the same
    # count, aggregated) — take the max so a transport that only exposes one
    # of the two signals (OpenRouter: searches is always [] per
    # krepis.llm_search, only usage.web_search_requests is populated) still
    # reports the real count instead of a false zero.
    n_searches = max(n_recorded, result.usage.web_search_requests)
    capture_llm_call(
        result,
        producer="morning_signal",
        sink_path=_config.EPISODES_DIR / f"{date_str}-{edition}.sft.jsonl",
        cost_usd=cost,
        meta={"date": date_str, "edition": edition,
              "recovery_pass": force_search},
    )
    return result, n_searches


def _coverage_recovery_directive(
    unmet: list[str],
    required_topics: list[dict],
    edition: str,
) -> str:
    """Build the escalated user-message addendum for a recovery regeneration.

    Names each uncovered segment plus its keyword figures (so the model knows
    exactly who to cover) and mandates a dedicated search + written segment for
    each. Appended to the original user message so the model still produces the
    COMPLETE edition, not just the missing segments (the script is a single
    header-less monologue — splicing fragments in is brittle).
    """
    by_name: dict[str, dict] = {}
    for t in required_topics:
        kws = [str(k) for k in (t.get("keywords") or []) if str(k).strip()]
        name = str(t.get("name") or ", ".join(kws))
        by_name[name] = t

    lines = []
    for name in unmet:
        kws = [str(k) for k in (by_name.get(name, {}).get("keywords") or []) if str(k).strip()]
        figures = ", ".join(kws)
        if figures:
            lines.append(
                f"  - {name}: run at least one dedicated web search covering "
                f"{figures}, then write its segment."
            )
        else:
            lines.append(
                f"  - {name}: run at least one dedicated web search, then "
                f"write its segment."
            )
    segments = "\n".join(lines)

    return (
        "\n\nCRITICAL COVERAGE CORRECTION. A prior draft of this exact "
        "edition OMITTED the following required segment(s) — they were not "
        "searched and not written. You MUST fix this now:\n"
        f"{segments}\n"
        "For EACH segment listed above: issue at least one dedicated web "
        "search for it BEFORE writing, and include a substantive spoken "
        "segment for it in today's script. Never write these from memory and "
        "never drop them. Produce the COMPLETE edition with every segment in "
        "order — not just the missing ones."
    )


def build_episode_request(
    config: dict,
    date_str: str,
    edition: str,
    schedule_entry: dict | None | object = _LOAD_SCHEDULE,
) -> dict:
    """Build everything :func:`generate_script` needs to issue its grounded
    LLM call, without calling the LLM.

    Factored out (config#1659 Phase B) so a second consumer can replay the
    EXACT same payload + guard configuration against a different
    ``ModelSpec`` without duplicating date/edition/schedule/news
    construction logic — used by ``scripts/oss_bakeoff.py`` (the shadow
    canary: same episode, prod anthropic spec vs. an openrouter candidate
    spec, candidate output never published/aired).

    Returns a dict: ``prompt_text``, ``user_content``, ``edition_label``,
    ``weekend``, ``effective_edition``, ``required_topics``,
    ``schedule_entry`` (the resolved value — may differ from the input
    when the sentinel triggered a load, or a self-loaded ``skip`` entry was
    downgraded to regular programming, matching :func:`generate_script`'s
    original resolution contract).
    """
    weekend = is_non_trading_day(date_str)
    prompt_text = load_prompt(weekend=weekend)

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    friendly_date = dt.strftime("%A, %B %-d, %Y")
    edition_label = "WEEKEND" if weekend else EDITION_LABELS[edition]
    opener = opening_line(edition, weekend)

    # Optional pre-fetched news context (config-gated, default OFF). When
    # enabled it is a HARD requirement by default: load_news_context RAISES
    # on a missing / malformed / stale / empty digest (run_date drives the
    # staleness check), aborting the pod before publish rather than
    # narrating yesterday's or no news. Set news_context.required: false to
    # fall back to fail-soft. When non-empty the block is injected BETWEEN
    # the edition sentence and the generate-instruction as a supplementary
    # reference; the canonical-opener instruction stays at the END.
    news_block = load_news_context(config, run_date=date_str)
    news_segment = f"{news_block}\n\n" if news_block else ""

    # Optional per-date scheduled content override (config-gated, default
    # OFF — see schedule_override.py). Fail-soft by design: any schedule
    # failure degrades to the regular episode (the loader WARNs + fires a
    # Telegram alert; the pod itself always ships). ``override`` devotes
    # the whole episode to the scheduled deep-dive topic; ``extend`` adds
    # one extra segment on top of the regular lineup. ``skip`` is handled
    # upstream in episode.main (a self-loaded skip degrades to regular
    # programming here — this function only generates).
    if schedule_entry is _LOAD_SCHEDULE:
        schedule_entry = load_schedule_override(config, date_str, edition)
    if schedule_entry and schedule_entry["mode"] == "skip":
        log.info(
            f"schedule: skip entry for {date_str} reached generate_script "
            f"(direct call?) — the skip decision belongs to episode.main; "
            f"generating regular programming"
        )
        schedule_entry = None

    schedule_segment = ""
    if schedule_entry:
        log.info(
            f"schedule: applying {schedule_entry['mode']} for {date_str}: "
            f"{schedule_entry['topic']}"
        )
        schedule_segment = (
            f"{format_schedule_directive(schedule_entry, edition_label)}\n\n"
        )

    if schedule_entry and schedule_entry["mode"] == "override":
        generate_instruction = (
            "Generate today's special deep-dive episode per the scheduled "
            "override above, keeping the system prompt's voice and format."
        )
    else:
        generate_instruction = (
            f"Generate today's {edition_label.lower()} episode per the "
            f"system prompt, respecting the News Window for this edition "
            f"(only news/events since the prior edition)."
        )

    user_content = (
        f"Today is {friendly_date}. This is the {edition_label} edition "
        f"of Morning Signal.\n\n"
        f"{news_segment}"
        f"{schedule_segment}"
        f"{generate_instruction}\n\n"
        f"Your response MUST begin verbatim with this exact line, "
        f"with no preamble or acknowledgement before it:\n\n"
        f"{opener}"
    )

    # required_topics construction mirrors the coverage-guard section below
    # in generate_script — see its comment block for the full rationale
    # (2026-06-17 tight-budget drop, 2026-06-29 blind-spot fix).
    required_topics = config.get("required_search_topics") or []
    if schedule_entry:
        guard = derived_topic_guard(schedule_entry)
        if schedule_entry["mode"] == "override":
            required_topics = [guard] if guard else []
        elif guard:
            required_topics = list(required_topics) + [guard]
    effective_edition = "weekend" if weekend else edition

    return {
        "prompt_text": prompt_text,
        "user_content": user_content,
        "opener": opener,
        "edition_label": edition_label,
        "weekend": weekend,
        "effective_edition": effective_edition,
        "required_topics": required_topics,
        "schedule_entry": schedule_entry,
    }


def _attempt_episode(
    llm_client: LLMClient,
    config: dict,
    prompt_text: str,
    user_content: str,
    date_str: str,
    edition: str,
    required_topics: list[dict],
    effective_edition: str,
    edition_label: str,
    friendly_date: str,
    force_search: bool = False,
) -> tuple[str, dict]:
    """Run one full generation attempt (first pass + optional self-heal
    recovery) against ``llm_client``'s spec. Returns ``(script, outcome)``
    where ``outcome`` is a small JSON-safe dict (provider/model/n_searches/
    unmet_topics/script_chars) for the LLM-decision log.

    Raises :exc:`RuntimeError` for the two HARD-abort policies —
    content-grounding verification and ``required_search_topics_fatal`` — same
    as always. :func:`generate_script` decides what a raise (or an empty-
    but-non-raising return) means: for a provider with no fallback
    configured, it propagates exactly as before; for a primary spec that HAS
    a fallback (config#1659, `fallback_llm` — a router group as of
    alpha-engine-config-I6980), it triggers a second attempt on the fallback
    spec instead.

    ``force_search=True`` deterministically forces the FIRST pass's
    ``web_search`` call via ``tool_choice`` (see ``_invoke_and_record``) —
    used by the per-segment self-heal recovery pass below so a dropped
    required-topic segment cannot fail with zero citations the way a
    discretionary (unforced) call can. Only the anthropic transport supports
    the force; callers must not pass ``force_search=True`` for an OpenRouter
    spec (``_invoke_and_record`` raises ``LLMConfigError`` there rather than
    silently degrading). ``generate_script`` no longer has a tier-3
    forced-search last resort of its own — retired with the direct-Anthropic
    cascade (alpha-engine-config-I6980); the force here is scoped to this
    recovery pass only.
    """
    result, n_searches = _invoke_and_record(
        llm_client, config, prompt_text, user_content, date_str, edition,
        force_search=force_search,
    )

    # Backstop against tool-call hallucination (see
    # _TOOL_CALL_HALLUCINATION_RE): a model instructed toward web_search but
    # given no tool can emit raw `<tool_calls>`/`<invoke name=...>` XML as
    # its entire response. This is unconditional (not just on the
    # non-grounded path below) — the content-grounding checks that would
    # normally catch a hollow script are themselves SKIPPED when
    # _news_context_grounded, which is exactly the condition under which
    # this failure mode occurs. Treated as a hard-abort of THIS attempt,
    # same as the content-grounding/required-topics raises below: it
    # triggers generate_script's fallback tier rather than airing garbage.
    if _TOOL_CALL_HALLUCINATION_RE.search(result.text):
        raise RuntimeError(
            f"{edition_label} edition for {friendly_date}: "
            f"provider={llm_client.spec.provider!r} model="
            f"{llm_client.spec.model!r} emitted an unexecuted tool call "
            f"instead of script content ({len(result.text)} chars): "
            f"{result.text[:200]!r} — refusing to publish"
        )

    # When the transport lacks server-side web search (litellm, openai,
    # direct), the pre-fetched news_context digest injected by
    # build_episode_request is the sole grounding source. Searches and
    # citations are both empty by construction — skip the web-search-
    # dependent guards (grounding verification, required_search_topics,
    # recovery pass) rather than falsely failing them.
    _news_context_grounded = (
        llm_client.spec.provider not in _PROVIDERS_WITH_WEB_SEARCH
    )

    unmet: list[str] = []
    if not _news_context_grounded:
        # Content-grounding verification: replaces the old raw search-count
        # floor (``min_web_searches``) which was a proxy metric — counting
        # tool invocations rather than measuring whether the model actually
        # used search content. Different providers search with very different
        # cadences, so a count floor penalizes providers that do fewer,
        # more targeted searches. The real signal is: did the search tool
        # return substantive web content? ``verify_script_grounded`` checks
        # that citations exist with real metadata, and also WARNs on
        # date-awareness gaps and anti-hallucination tells.
        # Raise BEFORE any TTS/publish so the silent-failure watchdog catches
        # the absent fresh episode instead of a bad one going live.
        min_citations = config.get("min_grounding_citations", 1)
        # Backward compat: if old ``min_web_searches`` is set, map it onto
        # the new key (0 = disable, just like before).
        if "min_web_searches" in config and "min_grounding_citations" not in config:
            min_citations = config["min_web_searches"]
            log.warning(
                "DEPRECATED: min_web_searches is retired in favour of "
                "min_grounding_citations (same semantics: ≥0; 0 = disable). "
                "Update your config."
            )
        if min_citations > 0:
            verify_script_grounded(
                script=result.text,
                citations=result.citations,
                date_str=date_str,
                edition_label=edition_label,
                friendly_date=friendly_date,
                provider=llm_client.spec.provider,
                min_citations=min_citations,
            )

        # Per-segment coverage guard: the global content-grounding check only
        # asserts the edition was grounded *somewhere*; it does NOT guarantee a
        # *specific* search-critical segment was covered. The failure mode this
        # catches (2026-06-17): with a tight ``web_search_max_uses`` budget the
        # model spends its searches on the earlier, digest-reinforced segments and
        # reaches the no-digest segments (e.g. a political pulse sourced only from
        # Truth Social / X) with no budget left — then writes them from memory.
        # ``required_search_topics`` lets the operator assert, per topic, that at
        # least ``min_matches`` searches actually targeted it. Default empty =
        # no-op (OSS-safe); the topics are declared in the operator's config
        # alongside the prompt that defines those segments. A topic can be scoped
        # to specific editions (``editions: [...]``) so a weekday-only segment does
        # not falsely abort the "weekend" edition, which runs a different prompt
        # with different segments and legitimately never searches it.
        #
        # Coverage handling has three tiers, in order of preference:
        #   1. SELF-HEAL (default) — if a required segment is uncovered, fire ONE
        #      bounded recovery regeneration whose user message names the dropped
        #      segment(s) and mandates a dedicated search + written segment for
        #      each. Adopt the recovered draft only if it covers strictly more.
        #      This actually FIXES the recurring political-segment drop (4th
        #      occurrence 2026-06-29) instead of only alerting on it.
        #   2. PUBLISH + ALERT (degraded-coverage policy, 2026-06-26, Brian) — if
        #      recovery is disabled or still can't cover the segment, an uncovered
        #      segment is a QUALITY defect, not grounds to withhold the whole
        #      episode: every OTHER segment is grounded, so a pod with one stale
        #      segment beats no pod. WARN log + flow-doctor/Telegram alert naming
        #      the uncovered topics, then publish. NOT a silent swallow (fail-loud
        #      policy): (a) the swallowed mode is a segment likely written from
        #      memory; (b) the episode still ships; (c) the surfaces are a WARN log
        #      + a Telegram alert.
        #   3. HARD ABORT — operators who would rather skip than ship a stale
        #      segment (the 2026-06-16 digest-hallucination posture) set
        #      ``required_search_topics_fatal: true``; recovery still runs first,
        #      so we abort only when even the forced retry could not cover it.
        # Coverage is judged with the SCRIPT in hand (not just search telemetry):
        # a topic counts as covered only when it was both searched AND its segment
        # actually aired — closing the blind spot where another segment's search
        # (e.g. "Elon Musk / SpaceX") falsely satisfied a political topic. The
        # global content-grounding check above stays in effect regardless — a
        # no-search-results edition is hallucinated wholesale, a different and
        # unrecoverable failure.
        if required_topics:
            fatal = config.get("required_search_topics_fatal", False)
            recover = config.get("required_search_topics_recover", True)
            # GroundedResult.text is the post-final-tool text; .searches carries
            # the per-query events (anthropic-only) and .citations carries the
            # cross-provider fallback (both extracted by krepis.llm_search) — no
            # response re-parsing needed.
            unmet = unmet_required_topics(
                result.searches, required_topics,
                edition=effective_edition, script=result.text,
                citations=result.citations,
            )

            if unmet and recover:
                log.warning(
                    f"DEGRADED COVERAGE on first pass for {edition_label} edition "
                    f"({friendly_date}): {', '.join(unmet)}. Firing one targeted "
                    f"recovery regeneration that forces these segment(s)."
                )
                directive = _coverage_recovery_directive(
                    unmet, required_topics, effective_edition
                )
                # Deterministically FORCE a web_search on the recovery pass rather
                # than only asking for one in prose — the prose ask is the same
                # stochastic compliance that already failed on the first pass.
                # (SearchOptions.force_first → forced server-tool tool_choice;
                # API-supported, verified live 2026-06-29. On a transport that
                # can't force it, _invoke_and_record degrades to prose-only
                # forcing and logs it — see its docstring.)
                try:
                    r2, n2 = _invoke_and_record(
                        llm_client, config, prompt_text, user_content + directive,
                        date_str, edition, force_search=True,
                    )
                    unmet2 = unmet_required_topics(
                        r2.searches, required_topics,
                        edition=effective_edition, script=r2.text,
                        citations=r2.citations,
                    )
                    if len(unmet2) < len(unmet):
                        log.info(
                            f"Recovery improved coverage for {edition_label} "
                            f"edition ({friendly_date}): {len(unmet)}→"
                            f"{len(unmet2)} uncovered "
                            f"(now: {', '.join(unmet2) or 'all covered'})."
                        )
                        result, n_searches, unmet = r2, n2, unmet2
                    else:
                        log.warning(
                            f"Recovery did not improve coverage for "
                            f"{edition_label} edition ({friendly_date}): still "
                            f"{len(unmet2)} uncovered ({', '.join(unmet2)}). "
                            f"Keeping the original draft."
                        )
                except Exception:  # noqa: BLE001 — recovery must never block publish
                    log.warning(
                        "Recovery regeneration failed; keeping the original "
                        "degraded draft and falling through to alert.",
                        exc_info=True,
                    )

            if unmet:
                if fatal:
                    log.error(
                        f"ABORT: {edition_label} edition for {friendly_date} did "
                        f"not cover required topic(s): {', '.join(unmet)} (even "
                        f"after recovery). The global search floor was met but a "
                        f"search-critical segment was skipped — almost certainly "
                        f"written from memory. required_search_topics_fatal=true → "
                        f"refusing to publish."
                    )
                    raise RuntimeError(
                        f"required search topic(s) not covered: {', '.join(unmet)} "
                        f"for {date_str}-{edition} — aborting before publish"
                    )
                budget = config.get("web_search_max_uses", 20)
                log.warning(
                    f"DEGRADED COVERAGE: {edition_label} edition for "
                    f"{friendly_date} shipped without covering required topic(s): "
                    f"{', '.join(unmet)} (budget web_search_max_uses={budget}). "
                    f"Publishing anyway + alerting for async triage. Set "
                    f"required_search_topics_fatal=true to hard-abort instead."
                )
                _alert_degraded_coverage(
                    config, effective_edition, edition_label, date_str,
                    unmet, n_searches, budget,
                )

    outcome = {
        "provider": result.provider,
        "model": result.model,
        "n_searches": n_searches,
        "unmet_topics": unmet,
        "script_chars": len(result.text),
    }
    return result.text, outcome


def _record_llm_decision(
    config: dict,
    date_str: str,
    edition: str,
    primary_spec: ModelSpec,
    used_spec: ModelSpec,
    fell_back: bool,
    primary_outcome: dict | None,
    fallback_outcome: dict | None,
) -> None:
    """Record which model actually produced (or failed to produce) the
    aired script — the operator-visible answer to "what LLM generated
    today's episode", independent of the granular per-call cost/search
    telemetry sinks :func:`_invoke_and_record` already writes.

    One JSON file per (date, edition), mirroring the existing
    ``{date}-{edition}.cost.jsonl``/``.searches.jsonl`` naming convention.
    Best-effort: a write failure is logged but never blocks publish, same
    policy as :func:`_alert_degraded_coverage` and ``record_applied``.

    Also best-effort synced to S3 (``LLM_DECISION_S3_PREFIX``) so the
    console's Morning Signal Schedule page can show which model aired each
    day without needing box access — same non-fatal policy as the local
    write; a sync failure never blocks publish or masks the local copy.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "date": date_str,
        "edition": edition,
        "primary_provider": primary_spec.provider,
        "primary_model": primary_spec.model,
        "used_provider": used_spec.provider,
        "used_model": used_spec.model,
        "fell_back": fell_back,
        "primary_outcome": primary_outcome,
        "fallback_outcome": fallback_outcome,
    }
    filename = f"{date_str}-{edition}.llm_decision.json"
    body = json.dumps(record, indent=2)

    path = _config.EPISODES_DIR / filename
    try:
        _config.EPISODES_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        log.info(
            f"LLM decision: {used_spec.provider}:{used_spec.model} "
            f"(fell_back={fell_back}) -> {path.name}"
        )
    except Exception:  # noqa: BLE001 — never block publish over a log write
        log.warning(
            "Failed to write LLM decision log (non-fatal)", exc_info=True
        )

    bucket = config.get("s3_bucket")
    if not bucket:
        return
    try:
        s3 = _aws_client("s3", region_name=config.get("s3_region", "us-west-2"))
        s3.put_object(
            Bucket=bucket,
            Key=f"{LLM_DECISION_S3_PREFIX}{filename}",
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        log.info(
            f"LLM decision synced to s3://{bucket}/{LLM_DECISION_S3_PREFIX}{filename}"
        )
    except Exception:  # noqa: BLE001 — never block publish over a sync failure
        log.warning(
            "Failed to sync LLM decision log to S3 (non-fatal, local copy "
            "still intact)", exc_info=True,
        )


def generate_script(
    config: dict,
    date_str: str,
    edition: str,
    schedule_entry: dict | None | object = _LOAD_SCHEDULE,
) -> str:
    """Call Claude with web search to generate the podcast script.

    ``schedule_entry``: the normalized per-date schedule entry from
    ``schedule_override.load_schedule_override``. ``episode.main`` loads
    the manifest once (for its skip guard) and passes the result through
    here; direct callers can omit it (the default sentinel loads it from
    config/S3, preserving the original call contract) or pass ``None``
    to explicitly run regular programming. ``skip`` entries never reach
    this function via the orchestrator; a self-loaded skip entry is
    treated as regular programming (the skip decision belongs to
    ``episode.main``, not here).

    Payload shape:
      - ``system``: static production prompt with ephemeral
        ``cache_control`` so the ~1.3K-token prefix hits the 0.1×
        cache-read rate on every tool-loop re-read inside one call.
      - ``messages[0]``: dynamic user preamble (date + edition + the
        opener instruction). The opener instruction lives in the user
        message, NOT the system block, so the static prefix stays
        cacheable per-call.
      - Post-process: if the response doesn't begin with the canonical
        opener, prepend it. Belt-and-suspenders for the case where the
        model emits a "Great, I now have enough info..." preamble.

    ``max_uses`` on ``web_search`` caps server-tool fees in the runaway
    case; 20 is above the empirical typical (~15).

    Provider fallback (config#1659, extended morning-signal-I118, retired to
    a compelled-route-only shape by alpha-engine-config-I6980): the ``llm``
    config knob can point at a router group (e.g. group ``high``). If the
    primary's OWN attempt hard-fails (content-grounding verification) or
    silently produces no usable content — a real, live-verified failure
    mode for reasoning-capable open-weight models, 2026-07-06 — this falls
    through to ``resolve_fallback_spec`` (config ``fallback_llm``), which
    MUST also name a router group and is resolved through the SAME
    ``_resolve_router_group`` path as the primary. `model-router-policy`
    §5.2 does not grant a fallback tier any more latitude than the primary
    to pick a provider by name, so neither one gets a direct-provider
    default: resolution failure here propagates and the episode fails
    closed (§5.3) rather than landing on a route nobody chose.

    There used to be a tier 3 — ONE forced-search attempt
    (``force_search=True``) on a hardcoded direct-Anthropic spec, added
    2026-07-16 after a compound failure lost an episode with no further
    tier to catch it. Retired here rather than converted to a router group:
    its entire value was the deterministic ``tool_choice`` force, which is
    an Anthropic-only transport capability (`_invoke_and_record` degrades it
    to best-effort or a no-op on every other transport) — coupling a
    grounding-quality guarantee to a specific provider is exactly the
    pattern this issue exists to end, and direct-Anthropic is a $0 budget
    line regardless (2026-07-17 ruling). A compound failure of both
    configured tiers now fails closed instead of getting a third attempt;
    :func:`_alert_fallback_exhausted` is what tells a human the episode
    didn't ship.

    Skipped when ``MORNING_SIGNAL_LLM`` pins an exact spec (the escape
    hatch means "run exactly this," not "with a hidden fallback"), and
    skipped when the primary IS ALREADY the anthropic spec (falling back to
    the same model fixes nothing and doubles cost — and a bare-anthropic
    primary can only happen via that same pinned escape hatch or the OSS
    self-host legacy default, neither of which this issue governs).
    :func:`_record_llm_decision` logs which model actually produced (or
    failed to produce) the script either way.
    """
    # A router group that will not resolve is a primary failure BEFORE any
    # call is made, not a reason to abort the episode: the configured fallback
    # is exactly what should carry it. The declared spec is kept so the
    # decision log still records what was asked for.
    primary_spec = declared_llm_spec(config)
    unresolvable_exc: Exception | None = None
    # Spec resolution can alert (a degraded route, model-router-policy §5.4)
    # and sits below the level that knows which edition is running, so the
    # edition is stamped here rather than threaded through two signatures that
    # have no other use for it.
    config[_EDITION_KEY] = edition
    try:
        primary_spec = resolve_llm_spec(config)
    except RouterGroupUnresolvable as exc:
        unresolvable_exc = exc

    req = build_episode_request(config, date_str, edition, schedule_entry)
    prompt_text = req["prompt_text"]
    user_content = req["user_content"]
    opener = req["opener"]
    edition_label = req["edition_label"]
    effective_edition = req["effective_edition"]
    required_topics = req["required_topics"]
    schedule_entry = req["schedule_entry"]

    friendly_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %-d, %Y")
    log.info(f"Generating {edition_label} script for {friendly_date}...")

    # `_COMPELLED_ROUTES` (above) makes krepis' own §5.2 degraded path
    # reachable, so the primary itself no longer falls through to a direct
    # provider on an edge-down degrade. This fallback tier is what remains
    # for a primary that resolved fine but whose ATTEMPT failed (content-
    # grounding verification, or a provider producing no usable content) —
    # `resolve_fallback_spec` now resolves `fallback_llm` through the same
    # router-group path (alpha-engine-config-I6980), so a total loss of the
    # router edge AND its registry-derived degraded route fails closed
    # (§5.3) via `_alert_fallback_exhausted` below, instead of publishing on
    # a route this repo chose. A total compelled-path loss means no episode
    # — that is the correct, intended outcome of this change, not a gap.
    fallback_eligible = (
        primary_spec.provider != "anthropic"
        and not os.environ.get(LLM_ENV_VAR)
    )

    primary_failed_exc: Exception | None = None
    if unresolvable_exc is not None:
        script, outcome = "", None
        primary_failed_exc = unresolvable_exc
    else:
        primary_client = LLMClient(primary_spec, callsite_id="morning-signal-episode-primary", max_retries=5)
        try:
            script, outcome = _attempt_episode(
                primary_client, config, prompt_text, user_content, date_str,
                edition, required_topics, effective_edition, edition_label,
                friendly_date,
            )
        except Exception as exc:
            script, outcome = "", None
            primary_failed_exc = exc

    used_spec = primary_spec
    fell_back = False
    fallback_outcome = None

    if not script and fallback_eligible:
        if primary_failed_exc is not None:
            log.error(
                f"{edition_label} edition for {friendly_date}: primary "
                f"provider={primary_spec.provider!r} model="
                f"{primary_spec.model!r} aborted ({primary_failed_exc}) — "
                f"falling back to the fallback spec."
            )
            if _is_deployment_class_failure(primary_failed_exc):
                _alert_primary_never_usable(
                    config, edition, primary_spec, primary_failed_exc,
                    friendly_date=friendly_date, edition_label=edition_label,
                )
        else:
            log.error(
                f"{edition_label} edition for {friendly_date}: primary "
                f"provider={primary_spec.provider!r} model="
                f"{primary_spec.model!r} produced no usable content after "
                f"its own retries — falling back to the fallback spec."
            )
        primary_failed_exc = None
        try:
            fallback_spec = resolve_fallback_spec(config)
            fallback_client = LLMClient(fallback_spec, callsite_id="morning-signal-episode-fallback", max_retries=5)
            script, fallback_outcome = _attempt_episode(
                fallback_client, config, prompt_text, user_content, date_str,
                edition, required_topics, effective_edition, edition_label,
                friendly_date,
            )
            used_spec = fallback_spec
            fell_back = True
        except Exception as exc:
            fallback_outcome = None
            primary_failed_exc = exc

        if not script:
            # Both compelled tiers are now exhausted and there is no direct-
            # provider tier 3 left to try (alpha-engine-config-I6980). §5.3:
            # fail closed rather than publish on a route this repo chose.
            log.error(
                f"{edition_label} edition for {friendly_date}: fallback "
                f"also failed ({primary_failed_exc}) — no further compelled "
                f"route to try. Failing closed (model-router-policy §5.3): "
                f"no episode today."
            )
            _alert_fallback_exhausted(
                config, edition, edition_label, friendly_date, primary_spec,
                primary_failed_exc,
            )

    if not script and primary_failed_exc is not None:
        # Either not fallback-eligible (anthropic-only config, or
        # MORNING_SIGNAL_LLM pinned) — original hard-abort behavior — or the
        # fallback tier was also exhausted, in which case
        # `_alert_fallback_exhausted` already fired above. primary_failed_exc
        # always holds the LAST tier's exception at this point.
        raise primary_failed_exc

    _record_llm_decision(
        config, date_str, edition, primary_spec, used_spec, fell_back,
        outcome, fallback_outcome,
    )

    if not script:
        log.error("Model returned no text content.")
        sys.exit(1)

    script = _scrub_preamble(script, opener)

    if not script.startswith(opener):
        log.warning(
            f"Response did not begin with canonical opener; prepending. "
            f"First 80 chars were: {script[:80]!r}"
        )
        script = f"{opener} {script}"

    word_count = len(script.split())
    log.info(f"Script: {len(script)} chars, ~{word_count} words (~{word_count / 150:.0f} min spoken)")

    # A schedule entry made it into the generated script — record the
    # best-effort applied marker (the console's ✅ badge; never raises,
    # never blocks TTS/publish).
    if schedule_entry:
        record_applied(config, date_str, edition, schedule_entry)

    return script


# (The post-final-tool text extraction that used to live here —
# `_final_text_after_last_tool`, the 2026-05-30 narration fix — was lifted
# verbatim into `krepis.llm_search.final_text_after_last_tool`; the adapter
# applies it when building `GroundedResult.text`.)


# HIGH-PRECISION meta-narration patterns. Since the post-final-tool extraction
# now removes pre/inter-search narration positionally (the primary defense),
# this regex pass is a BACKSTOP — so it must err toward NOT matching. Each
# pattern pairs a first-person / process cue with an explicit search/segment
# object, so legitimate spoken copy that merely opens with "Let's…", "Great…",
# or mentions "search results" as a news fact is preserved (2026-05-30:
# "Let's start with the numbers." and "Great news from the cosmos this week."
# were false-positived by the older bare-opener patterns).
_META_PREAMBLE_LINE_PATTERNS = [
    # First-person pronoun + a process verb/object ("Let me search…", "I need to
    # gather…", "I'll pull the latest…", "Let me craft the segment"). Bare
    # "Let's start with…" / "Let me walk you through…" do NOT match — no process cue.
    re.compile(
        r"^\s*(I'll|I will|I need to|I need more|I'm going to|I am going to|Let me|Let's|Now let me|First,?\s+let me)\b"
        r".{0,40}\b(search|searching|gather|gathering|compile|compiling|research(ing)?|look(ing)?\s+up|fetch|pull(ing)?|find\b|the segment|the coverage|the briefing|enough info|more info|the latest|fresh info)",
        re.IGNORECASE,
    ),
    # Bare acknowledgement that the model emits before "delivering" — REQUIRES
    # trailing punctuation ("Perfect.", "Great,") so "Great news…" / "Perfect
    # storm…" (legit copy) are NOT matched.
    re.compile(r"^\s*(Great|Sure|Okay|OK|Alright|Got it|Perfect|Excellent)[,.!:]\s", re.IGNORECASE),
    # "Here's the X segment/briefing/edition" — production terms unlikely in
    # content. Dropped "update"/"coverage" objects ("Here's the update on…" is
    # plausible news copy).
    re.compile(r"^\s*Here(\s+is|'s)\s+(your|the|today's)\s+(edition|briefing|episode|segment)\b", re.IGNORECASE),
    # A LINE that LEADS with a search/gather process verb + freshness noun
    # ("Searching for the latest news…"). Anchored + no "research" so content
    # like "Investors search for the latest yield data" or "Research on recent
    # data shows…" (mid-sentence / research-as-topic) is preserved.
    re.compile(r"^\s*(I\s+)?(gather|gathering|search(ing)?\s+for|look(ing)?\s+up|fetch|pull(ing)?\s+up)\b.{0,40}\b(fresh|latest|current|recent|information|news|data|info|updates?|developments?)\b", re.IGNORECASE),
    # "I now have enough/comprehensive … " (kept narrow — drops the loose "the"/"good"/"all the").
    re.compile(r"\bnow\s+have\s+(enough|comprehensive|sufficient|solid)\b", re.IGNORECASE),
    # The model narrating its OWN search results — must LEAD the line and use a
    # strong reporting verb, so content that merely mentions "search results"
    # mid-sentence (e.g. "Users seeking traditional search results have…", a
    # real tech-segment topic) is preserved.
    re.compile(r"^\s*(the\s+)?search\s+results?\s+(show|indicate|reveal|confirm|suggest|point\s+to)\b", re.IGNORECASE),
    # "Based on/According to/Looking at … search …" framing.
    re.compile(r"^\s*(Based on|According to|Looking at|From|Drawing on)\b.{0,40}\bsearch(es|ing)?\b", re.IGNORECASE),
    # A LINE that LEADS with "craft/compile/put together … the segment/briefing"
    # (first-person "Let me craft the segment" is already caught above). Anchored
    # + dropped "write"/"coverage"/"edition" so "writers prepare coverage of the
    # election" (content) is preserved.
    re.compile(r"^\s*(I\s+)?(craft|crafting|compile|compiling|prepare|preparing|put\s+together)\b.{0,40}\b(segment|briefing|episode)\b", re.IGNORECASE),
    # "need (more|additional|fresher) current/recent information/coverage".
    re.compile(r"\bneed\s+(more|additional|fresher?|the\s+latest)\b.{0,30}\b(current|recent|up-to-date|information|data|coverage|news)\b", re.IGNORECASE),
]


def _scrub_preamble(script: str, opener: str) -> str:
    """Strip leading meta-narration before the canonical opener.

    Belt-and-suspenders for cases where the model emits lines like
    "I'll gather fresh info on..." or "Let me search for..." before
    launching into the podcast. The prompt forbids these but can't
    strictly enforce from the model side — strip post-hoc so the TTS
    never sees them.

    Strategy: if the canonical opener appears within the first 800
    chars, drop everything before it; otherwise drop leading paragraphs
    whose first non-blank line matches a known meta-preamble pattern.
    """
    if not script:
        return script

    head_window = script[:800]
    idx = head_window.find(opener)
    if idx > 0:
        log.warning(
            f"Scrubbing {idx} chars of preamble before opener: "
            f"{script[:idx][:160]!r}"
        )
        return script[idx:].lstrip()

    paragraphs = script.split("\n\n")
    while paragraphs:
        first = paragraphs[0].strip()
        if not first:
            paragraphs.pop(0)
            continue
        first_line = first.split("\n", 1)[0]
        if any(p.search(first_line) for p in _META_PREAMBLE_LINE_PATTERNS):
            log.warning(f"Scrubbing meta-preamble paragraph: {first[:160]!r}")
            paragraphs.pop(0)
            continue
        break

    scrubbed = "\n\n".join(paragraphs).strip()
    if not scrubbed:
        log.warning(
            "Scrub would empty script; returning original so the "
            "opener-prepend fallback can rescue. Original: "
            f"{script[:160]!r}"
        )
        return script
    return scrubbed
