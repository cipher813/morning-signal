"""Payload-shape smoke check — catches Anthropic payload-shape regressions
that mocked unit tests miss by design, WITHOUT dispatching a live call.

The unit-test suite uses ``MagicMock`` to stand in for
``anthropic.Anthropic().messages.create()`` so tests run offline and
cheaply, but that means the suite never independently re-validated the
payload the krepis producer-side chokepoint already checks. The 2026-05-26
incident (HTTP 400 "This model does not support assistant message prefill.
The conversation must end with a user message.") slipped past CI because
mocked tests can't see the server-tool ⊥ assistant-prefill constraint.

Retired the live network dispatch here on 2026-08-29 (Brian ruling: "we
shouldn't be using the anthropic api at all" — the direct-Anthropic API
budget is $0; morning-signal-I165 / model-router-policy). The live call's
marginal value over a static check was narrow — proving the ANTHROPIC
SERVER also accepts the shape, on top of the shape already being
schema-valid — and not worth funding an Anthropic API key + a per-PR
network dependency to keep.

``krepis.anthropic_payload.build_messages_payload`` already runs
``validate_payload`` at construction time (the SAME producer-side
chokepoint ``generate_script`` goes through via the krepis ``LLMClient``
anthropic transport), enforcing exactly the invariant the 2026-05-26
incident needed:

1. server-tool ⊥ trailing-assistant-message (prefill) — the incident class.
2. the 4-``cache_control``-breakpoint ceiling.

This script builds the SAME payload shape ``generate_script`` produces
(server-tool + cached system block + single user message, no prefill) and
calls ``validate_payload`` explicitly, so the check fails loud and
specifically even if a future krepis version stops validating at
construction time. Deterministic, free, and needs no credential — every PR
runs it, forks included, with nothing to skip.

Designed to run:

  * In CI on every PR that touches ``src/morning_signal/claude.py``,
    ``prompt.md``, ``config.yaml*``, or ``pyproject.toml``.
  * Locally, via ``.venv/bin/python tests/live_api_smoke.py`` — no env
    vars required.

Stays out of pytest's default collection because the filename doesn't
match ``test_*.py``. That's intentional — pytest runs offline always;
this script is invoked explicitly by CI (and can be by an operator).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `morning_signal` importable when the script is run directly via
# `python tests/live_api_smoke.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from krepis.anthropic_payload import (  # noqa: E402
    PayloadInvariantError,
    build_messages_payload,
    build_web_search_tool,
    validate_payload,
)
from morning_signal.claude import (  # noqa: E402
    EDITION_LABELS,
    opening_line,
)

SMOKE_MODEL = "claude-sonnet-4-6"
SMOKE_SYSTEM_PROMPT = (
    "You are a podcast script writer for Morning Signal. This is a CI "
    "payload-shape smoke test — respond with one word."
)


def build_smoke_payload() -> dict:
    """Build a payload with the SAME shape ``generate_script`` produces:
    server-tool (``web_search_20250305`` + ``max_uses=20``), cached
    system block, single user message with the opener instruction
    embedded, NO assistant prefill. Routes through
    ``krepis.anthropic_payload.build_messages_payload``, which already
    runs ``validate_payload`` at construction time — the explicit call in
    :func:`main` re-asserts it so this smoke's protection does not
    silently depend on that internal behavior never changing.
    """
    edition = "am"
    weekend = False
    opener = opening_line(edition, weekend)
    edition_label = EDITION_LABELS[edition]

    tools = [build_web_search_tool(max_uses=20)]
    user_content = (
        f"This is the {edition_label} edition of Morning Signal "
        f"(CI smoke).\n\n"
        f"Your response MUST begin verbatim with this exact line, "
        f"with no preamble:\n\n{opener}"
    )
    return build_messages_payload(
        model=SMOKE_MODEL,
        system_prompt=SMOKE_SYSTEM_PROMPT,
        user_content=user_content,
        max_tokens=1,
        tools=tools,
        cache_system=True,
    )


def main() -> int:
    print(
        "live_api_smoke: building the production payload shape (no "
        "network call, no credential) ...",
        file=sys.stderr,
    )
    try:
        payload = build_smoke_payload()
        validate_payload(payload)
    except PayloadInvariantError as exc:
        print(
            f"live_api_smoke: FAILED — payload shape is invalid.\n"
            f"  Error: {exc}\n"
            f"  This is exactly the regression class this smoke is meant "
            f"to catch (see the 2026-05-26 incident in the module "
            f"docstring). DO NOT MERGE.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            f"live_api_smoke: unexpected error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        "live_api_smoke: OK — payload shape valid "
        f"(model={payload['model']!r}, tools={len(payload.get('tools') or [])}, "
        f"messages={len(payload['messages'])})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
