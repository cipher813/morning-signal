"""Operational canary — payload-shape regression gate for morning-signal.

Sibling to ``tests/live_api_smoke.py`` (CI gate at PR time) but designed
to run on the EC2 host as the systemd ``ExecStartPre=`` for the
``morning-signal.service`` unit. Catches the long-tail regression class
that CI's paths-filter cannot see: out-of-band edits to the LIVE
production prompt + config that bypass the PR flow.

Examples this catches that CI does not:
  - operator edits ``prompt.md`` / ``prompt_weekend.md`` directly on the
    host then ``git commit --no-verify`` push that misses CI;
  - operator edits the SSM ``/morning-signal/config-yaml`` parameter or
    the S3-hosted prompt object (per
    ``reference_morning_signal_prompts_via_s3_260527``) without a PR;
  - lib-pin bumps via ``pip install`` overrides on the host that don't
    update ``pyproject.toml``.

Behavior: loads the EXACT production config + prompts the next
``generate_script`` call would use (via the same ``_maybe_load_from_ssm``
bootstrap), resolves the SAME spec ``generate_script``'s primary would
(``morning_signal.claude.resolve_llm_spec`` — through the krepis router
when ``llm`` names a group, refusing anything outside
``_COMPELLED_ROUTES``), and dispatches a single ``max_tokens=1`` call
through ``krepis.llm.LLMClient`` (~$0.001), and exits 0/1.

Routed through krepis rather than a direct ``anthropic.Anthropic`` client
as of alpha-engine-config-I6980 — this script previously bypassed the
router entirely, which is a direct-provider call this consumer is not
permitted to make (`model-router-policy` R26).

Exit codes:
  0 — request validated by ``krepis.llm.LLMClient`` (server-tool ⊥
      assistant-prefill shape included) AND accepted by the resolved
      provider at runtime.
  1 — spec resolution failure (the router group did not resolve —
      generate_script's primary would fail identically), request/shape
      validation failure, an API-level rejection, or any unexpected error.

This is Phase A — the script itself. Phase B is wiring it into the
``morning-signal.service`` unit as an ``ExecStartPre=`` so a payload-shape
regression blocks the service from starting rather than failing at the next
scheduled run.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import sys
from pathlib import Path

# Make ``morning_signal`` importable when the script is run directly via
# ``python scripts/canary.py`` from the repo root (or via
# ``.venv/bin/python scripts/canary.py`` from the systemd unit).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from krepis.llm import LLMClient, SearchOptions  # noqa: E402
from krepis.llm_config import LLMConfigError  # noqa: E402
from morning_signal import aws as _aws  # noqa: E402
from morning_signal import config as _config  # noqa: E402
from morning_signal.aws import _maybe_load_from_ssm  # noqa: E402
from morning_signal.claude import (  # noqa: E402
    EDITION_LABELS,
    RouterGroupUnresolvable,
    _PROVIDERS_WITH_WEB_SEARCH,
    is_non_trading_day,
    opening_line,
    resolve_llm_spec,
)
from morning_signal.config import load_config, load_prompt  # noqa: E402

log = logging.getLogger("morning-signal.canary")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _build_canary_request(
    date_str: str,
    edition: str,
) -> tuple[str, str]:
    """Construct the SAME ``(system, user_content)`` pair ``generate_script``
    builds — everything that varies with date/edition, so any payload-shape
    regression there is caught here too.

    The payload itself (cache-control, server-tool ⊥ assistant-prefill
    shape, tool-call shape) is now built and validated by
    ``krepis.llm.LLMClient`` at dispatch time — the SAME code path
    ``generate_script`` calls through, rather than a duplicate local build.
    Routing through ``LLMClient`` (alpha-engine-config-I6980) is what makes
    that true: this script previously constructed the Anthropic payload with
    a hardcoded ``claude_model`` and dispatched it with a bare
    ``anthropic.Anthropic`` client, which is a direct-provider call this
    consumer is not permitted to make — see ``_COMPELLED_ROUTES`` in
    ``morning_signal.claude``.
    """
    weekend = is_non_trading_day(date_str)
    prompt_text = load_prompt(weekend=weekend)

    dt = _dt.datetime.strptime(date_str, "%Y-%m-%d")
    friendly_date = dt.strftime("%A, %B %-d, %Y")
    edition_label = "WEEKEND" if weekend else EDITION_LABELS[edition]
    opener = opening_line(edition, weekend)

    user_content = (
        f"Today is {friendly_date}. This is the {edition_label} edition "
        f"of Morning Signal. Generate today's "
        f"{edition_label.lower()} episode per the system prompt, respecting "
        f"the News Window for this edition (only news/events since the "
        f"prior edition).\n\n"
        f"Your response MUST begin verbatim with this exact line, "
        f"with no preamble or acknowledgement before it:\n\n"
        f"{opener}"
    )

    return prompt_text, user_content


def main() -> int:
    try:
        # Mirrors episode.py/cli.py's bootstrap order exactly: assume the
        # runner role BEFORE touching SSM/S3. Missing this step silently
        # falls back to the box's own EC2 instance-profile credentials,
        # which have no S3 grant on morning-signal-podcast at all — masked
        # for months by that bucket's public-read policy (fixed
        # 2026-07-06, PR #104) until oss_bakeoff.py's own verification run
        # surfaced the identical gap as an AccessDenied on prompts/prompt.md.
        _aws._AWS_SESSION = _aws._load_runner_session()
        _maybe_load_from_ssm()
    except Exception as exc:
        log.error(
            "canary: SSM bootstrap failed (%s: %s). The production "
            "service would fail the same way; refusing to release the "
            "service to ExecStart.",
            type(exc).__name__,
            exc,
        )
        return 1

    try:
        cfg = load_config()
    except SystemExit:
        log.error("canary: load_config() exited; config.yaml missing or "
                  "unreadable. See preceding log line for path.")
        return 1
    except Exception as exc:
        log.error(
            "canary: load_config() raised (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return 1

    today = _dt.date.today().isoformat()
    edition = os.environ.get("MORNING_SIGNAL_CANARY_EDITION", "am")
    if edition not in EDITION_LABELS:
        log.error(
            "canary: MORNING_SIGNAL_CANARY_EDITION=%r is not one of %s",
            edition,
            sorted(EDITION_LABELS),
        )
        return 1

    try:
        prompt_text, user_content = _build_canary_request(today, edition)
    except Exception as exc:
        log.error(
            "canary: request construction failed (%s: %s) — this would "
            "fail the same way in generate_script. DO NOT START.",
            type(exc).__name__,
            exc,
        )
        return 1

    # Resolve the SAME spec generate_script's primary would resolve —
    # including, when ``llm`` names a router group, walking the router
    # (``_resolve_router_group``) and refusing anything outside
    # ``_COMPELLED_ROUTES``. A router group that will not resolve from this
    # deployment is exactly the "DO NOT START" condition this canary exists
    # to catch (alpha-engine-config-I6980: this script previously bypassed
    # the router entirely and dispatched straight to the Anthropic API,
    # which is a direct-provider call this consumer is not permitted to
    # make).
    try:
        spec = resolve_llm_spec(cfg)
    except RouterGroupUnresolvable as exc:
        log.error(
            "canary: FAILED — the configured router group did not resolve "
            "from this deployment (%s). generate_script's primary would "
            "fail identically. DO NOT START.",
            exc,
        )
        return 1
    except Exception as exc:
        log.error(
            "canary: spec resolution raised (%s: %s). DO NOT START.",
            type(exc).__name__,
            exc,
        )
        return 1

    llm_client = LLMClient(spec, callsite_id="morning-signal-canary", max_retries=0)

    log.info(
        "canary: dispatching max_tokens=1 smoke to provider=%s model=%s "
        "(edition=%s, config=%s)",
        spec.provider,
        spec.model,
        edition,
        _config.CONFIG_FILE,
    )

    try:
        if spec.provider in _PROVIDERS_WITH_WEB_SEARCH:
            result = llm_client.complete_grounded(
                system=prompt_text,
                user_content=user_content,
                search=SearchOptions(
                    max_uses=cfg.get("web_search_max_uses", 20),
                    force_first=False,
                ),
                max_tokens=1,
                cache_system=True,
            )
        else:
            result = llm_client.complete(
                system=prompt_text,
                user_content=user_content,
                max_tokens=1,
                cache_system=True,
                on_unsupported="drop",
            )
    except LLMConfigError as exc:
        log.error(
            "canary: FAILED — %s resolved to provider=%s, which krepis "
            "rejected as a shape/config mismatch: %s. This is the exact "
            "regression class the canary is meant to catch (see ROADMAP "
            "L380; the 2026-05-26 server-tool ⊥ assistant-prefill "
            "incident, and the router-group payload-shape regression this "
            "canary now also covers). DO NOT START the service.",
            spec.model,
            spec.provider,
            exc,
        )
        return 1
    except Exception as exc:
        log.error(
            "canary: unexpected error (%s: %s). DO NOT START.",
            type(exc).__name__,
            exc,
        )
        return 1

    log.info(
        "canary: OK — provider=%s model=%s usage=%s",
        result.provider,
        result.model,
        result.usage,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
