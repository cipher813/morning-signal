"""Shadow-canary bakeoff: prod (Anthropic) vs. Phase-B OSS candidates
(OpenRouter) coverage-guard parity — config#1659, scope item 5.

Brian's ratified destination for morning-signal (2026-07-03) is an
open-weight model via OpenRouter + the ``openrouter:web_search`` server
tool. The live ``llm`` config/SSM flip is gated (No-Shortcuts) on this
script's evidence: three production incident-guards (``min_web_searches``,
``required_search_topics``, forced-search recovery) were re-keyed to work
off whichever signal a transport actually exposes (see ``claude.py`` and
``search_telemetry.py``), but that re-key needs to be PROVEN safe against
real candidate responses before it governs a live edition — this script is
that proof.

Two candidates run side by side against the same prompt (2026-07-06,
Artificial Analysis Intelligence Index — both tie for #1 among open-weight
models): ``moonshotai/kimi-k2.6`` (the original config#1659 pick, also the
top open-weight model for agentic/tool-use benchmarks) and
``xiaomi/mimo-v2.5-pro`` (ties Kimi on general intelligence, ~4x cheaper on
completion tokens, 1M vs 256K context). Both are reasoning models and both
carry ``reasoning: {"exclude": true}`` (krepis>=0.11.0,
``ModelSpec.reasoning``) — without it, a reasoning-capable model can spend
its entire output budget on invisible chain-of-thought and return an
empty ``message.content`` even at a generous ``max_tokens`` (reproduced
live 2026-07-06 against Kimi K2.6 with the real production prompt:
``finish_reason="stop"``, ~15K reasoning chars, ~1 char of actual content).

For a given (date, edition) it builds ONE shared prompt + guard
configuration via ``claude.build_episode_request`` (so every side sees the
EXACT same system prompt, user message, and ``required_search_topics`` the
real production run would use), then issues one grounded call per side —
the current production spec (``resolve_llm_spec``, still Anthropic per
Phase A) plus one per candidate — and records a parity comparison to a
JSONL log. NO side is published or TTS'd; this never touches
``episode.py``, the RSS feed, or ``_config.EPISODES_DIR`` (the real
episode's telemetry sinks) — it is a side-channel measurement only.

Run daily (cron/systemd timer, alongside the real production pipeline) for
the ≥2-week bakeoff window. Once a candidate's ``unmet_topics`` matches
prod for ≥2 weeks straight (config#1659's closes-when criterion), the live
``llm`` flip can be scheduled with an operator confident the coverage
guards hold on that candidate.

Each run's JSONL record is written locally (``bakeoff_logs/`` by default)
AND best-effort synced to ``s3://{config[s3_bucket]}/ops/bakeoff/`` for
durability across a box replacement — that prefix is private (2026-07-06:
the bucket's public-read policy was tightened from a bucket-wide wildcard
to exactly ``episodes/*``/``feed.xml``/``artwork.jpg``, so anything else
written here, including this prefix, is authenticated-only by the
policy's own absence).

Usage::

    python scripts/oss_bakeoff.py --date 2026-07-06 --edition am

Every run issues real, BILLED grounded calls — one per side. It therefore
refuses to run within ``--min-interval-days`` (default 5) of the last
completed comparison, whatever started it, and says so; ``--force`` is the
deliberate operator re-run. That is a minimum interval and not a
day-of-week window on purpose: a window would also refuse the legitimate
late run ``Persistent=true`` replays after the box was down at the
scheduled moment (alpha-engine-config-I9000).

Exit codes: 0 on a completed comparison (regardless of parity outcome — a
mismatch is exactly what this script exists to surface, not an error) AND
on a run refused by the interval guard (an off-schedule launch is not a
bakeoff failure, and exiting non-zero would manufacture a red unit and a
page); 1 on a setup/run failure (missing OPENROUTER_API_KEY, SSM bootstrap
failure, LLM call failure on any side).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import sys
from pathlib import Path

# Make ``morning_signal`` importable when run directly via
# ``python scripts/oss_bakeoff.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from krepis.llm import LLMClient  # noqa: E402
from krepis.llm_config import ModelSpec  # noqa: E402

from morning_signal import aws as _aws  # noqa: E402
from morning_signal.aws import _aws_client, _maybe_load_from_ssm  # noqa: E402
from morning_signal.claude import (  # noqa: E402
    build_episode_request,
    call_with_grounding_degrade,
    resolve_llm_spec,
)
from morning_signal.config import load_config  # noqa: E402
from morning_signal.search_telemetry import unmet_required_topics  # noqa: E402

log = logging.getLogger("morning-signal.oss_bakeoff")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# The Phase B candidate pool (config#1659). Not config-driven — this
# script's whole job is generating the evidence that eventually picks ONE
# of these (or neither) for the live ``llm`` flip value. ``reasoning``:
# see the module docstring — both are reasoning models and need this to
# avoid the empty-content failure mode found 2026-07-06.
CANDIDATES = [
    {"label": "kimi-k2.6", "model": "moonshotai/kimi-k2.6", "reasoning": {"exclude": True}},
    {"label": "mimo-v2.5-pro", "model": "xiaomi/mimo-v2.5-pro", "reasoning": {"exclude": True}},
]

# Where bakeoff JSONL records land. Deliberately separate from
# _config.EPISODES_DIR (the real production episode's telemetry sinks) —
# this is a side-channel measurement log, never mixed with aired-episode
# data. Overridable for the systemd unit / local runs.
BAKEOFF_LOG_DIR_ENV = "MORNING_SIGNAL_BAKEOFF_LOG_DIR"
DEFAULT_BAKEOFF_LOG_DIR = "bakeoff_logs"

# S3 durability: local box disk alone doesn't survive a box replacement
# across the ≥2-week bakeoff window, so each run also uploads to the
# product's OWN bucket (not the shared alpha-engine-research bucket — the
# morning-signal-runner IAM role has no write grant there, only on its own
# morning-signal-podcast bucket). This prefix is PRIVATE: the bucket's
# public-read policy (2026-07-06 fix, was a bucket-wide wildcard that also
# leaked the proprietary prompts/ + schedule/ prefixes) is scoped to
# EXACTLY episodes/*, feed.xml, and artwork.jpg — ops/bakeoff/ isn't
# listed, so it stays authenticated-only by the policy's own absence, not
# by convention. Runner role already has bucket-wide PutObject, so no new
# IAM grant is needed for this prefix.
BAKEOFF_S3_PREFIX = "ops/bakeoff/"

# MINIMUM INTERVAL between two billed comparisons — defence in depth for a
# workload that spends real money (alpha-engine-config-I9000).
#
# On 2026-08-28 this script ran at 03:01 UTC on a Friday. Its timer had not
# elapsed and would not until the following Wednesday: an unrelated
# crucible-dashboard deploy re-ran the systemd installer, and `Requires=` in the
# timer's [Unit] made `systemctl enable --now <timer>` start the service as a
# dependency. That launcher defect is fixed in crucible-dashboard, and this
# guard exists because a unit that costs money per run must not rely on its
# launcher being correct — any launcher, on any box, including one this repo
# does not own (self-hosters run their own).
#
# It is a MINIMUM INTERVAL, deliberately not a day-of-week window. A window
# would also refuse the legitimate late run that `Persistent=true` replays after
# the box was down at the scheduled moment — trading one failure mode for
# another. An interval refuses only what a correct weekly cadence would never
# produce: a second billed comparison a few days after the last one.
#
# Five days, not seven: a weekly timer whose slot was missed can legitimately
# fire a day or two late, and the guard must not eat that run. It still refuses
# same-day and next-day repeats, which is the whole observed failure.
DEFAULT_MIN_INTERVAL_DAYS = 5

# ``{YYYY-MM-DD}-{edition}.bakeoff.jsonl`` — written only after a comparison
# COMPLETES, so a setup failure leaves no marker and its retry is never blocked.
_LOG_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(?:am|pm)\.bakeoff\.jsonl$")


def _last_bakeoff_date(log_dir: Path) -> "_dt.date | None":
    """Most recent date a comparison completed for, or None if never."""
    dates = []
    try:
        names = [p.name for p in log_dir.iterdir()]
    except OSError:
        return None
    for name in names:
        match = _LOG_NAME_RE.match(name)
        if match:
            try:
                dates.append(_dt.date.fromisoformat(match.group(1)))
            except ValueError:  # pragma: no cover - regex already pins the shape
                continue
    return max(dates) if dates else None


def _run_side(
    *,
    label: str,
    spec: ModelSpec,
    config: dict,
    prompt_text: str,
    user_content: str,
    required_topics: list[dict],
    effective_edition: str,
) -> dict:
    """Issue one generation call on ``spec`` and score it against the SAME
    coverage guards the production path enforces (see ``claude.py``).

    The call goes through ``claude.call_with_grounding_degrade`` — the exact
    helper the live episode calls — rather than a local ``complete_grounded``.
    The local copy lacked production's degrade-to-``complete()`` branch for a
    transport with no server-side web search, so when the production ``llm``
    spec moved to a krepis router group (#135, merged 2026-08-02) the prod
    side of this comparison began raising ``LLMConfigError`` — exit 1 on every
    weekly run from 2026-08-05 — while production itself kept airing. A shadow
    bakeoff that cannot exercise the spec production actually uses measures
    nothing.

    What is deliberately NOT shared is ``_invoke_and_record``'s telemetry:
    that writes to ``episodes/{date}-{edition}.cost.jsonl`` and the episode's
    search/SFT sinks, and these calls are never published. Mixing them into
    the aired episode's billed-cost record is precisely what
    ``BAKEOFF_LOG_DIR_ENV`` exists to prevent.
    """
    client = LLMClient(spec, callsite_id="morning-signal-oss-bakeoff", max_retries=3)
    result = call_with_grounding_degrade(client, config, prompt_text, user_content)
    # Provider-agnostic search count — mirrors claude._invoke_and_record.
    n_searches = max(len(result.searches), result.usage.web_search_requests)
    unmet = unmet_required_topics(
        result.searches, required_topics,
        edition=effective_edition, script=result.text,
        citations=result.citations,
    )
    min_citations = config.get("min_grounding_citations", 1)
    return {
        "label": label,
        "provider": result.provider,
        "model": result.model,
        "n_searches": n_searches,
        "n_citations": len(result.citations),
        "has_grounding_citations": len(result.citations) >= min_citations,
        "unmet_topics": unmet,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "provider_cost_usd": result.usage.provider_cost_usd,
        "script_chars": len(result.text),
        "script_words": len(result.text.split()),
        # Full text kept for qualitative side-by-side review (config#1659
        # scope item 5: "compare ... script quality") — this log is never
        # published, so storing it here is safe.
        "script_text": result.text,
    }


def _parity(prod: dict, candidate: dict) -> dict:
    return {
        "both_met_grounding": (
            prod["has_grounding_citations"] and candidate["has_grounding_citations"]
        ),
        "unmet_topics_match": (
            set(prod["unmet_topics"]) == set(candidate["unmet_topics"])
        ),
        # The gate this script exists to catch: candidate silently
        # covering FEWER required topics than prod would on the same
        # prompt/config. Equal or better is fine; strictly worse is the
        # signal that keeps the live flip gated.
        "candidate_strictly_worse": (
            len(candidate["unmet_topics"]) > len(prod["unmet_topics"])
        ),
    }


def run_bakeoff(config: dict, date_str: str, edition: str) -> dict:
    """Build the shared episode request once, run prod + every candidate,
    return the parity comparison record (also written to the JSONL log by
    ``main``).
    """
    req = build_episode_request(config, date_str, edition)
    prod_spec = resolve_llm_spec(config)

    prod = _run_side(
        label="prod", spec=prod_spec, config=config,
        prompt_text=req["prompt_text"], user_content=req["user_content"],
        required_topics=req["required_topics"],
        effective_edition=req["effective_edition"],
    )

    candidates: dict = {}
    for c in CANDIDATES:
        spec = ModelSpec(
            "openrouter", c["model"],
            max_tokens=config.get("max_tokens", 4096),
            reasoning=c.get("reasoning"),
        )
        result = _run_side(
            label=c["label"], spec=spec, config=config,
            prompt_text=req["prompt_text"], user_content=req["user_content"],
            required_topics=req["required_topics"],
            effective_edition=req["effective_edition"],
        )
        candidates[c["label"]] = {
            **result,
            "parity": _parity(prod, result),
        }

    return {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "date": date_str,
        "edition": edition,
        "required_topic_names": [
            str(t.get("name") or ", ".join(t.get("keywords") or []))
            for t in req["required_topics"]
        ],
        "prod": prod,
        "candidates": candidates,
    }


def _sync_to_s3(config: dict, local_path: Path, date_str: str, edition: str) -> None:
    """Best-effort upload of the day's bakeoff JSONL to S3 for durability
    across the ≥2-week window — local box disk alone doesn't survive a box
    replacement. Secondary to the local write (which already succeeded by
    the time this runs), so a failure here is logged loudly but never
    crashes the run — the comparison result itself is unaffected.
    """
    bucket = config.get("s3_bucket")
    if not bucket:
        log.warning(
            "bakeoff: no s3_bucket in config — skipping S3 sync for %s-%s "
            "(local copy at %s is the only record).",
            date_str, edition, local_path,
        )
        return
    region = config.get("s3_region", "us-west-2")
    s3_key = f"{BAKEOFF_S3_PREFIX}{local_path.name}"
    try:
        s3 = _aws_client("s3", region_name=region)
        s3.upload_file(
            str(local_path), bucket, s3_key,
            ExtraArgs={"ContentType": "application/x-ndjson"},
        )
        log.info("bakeoff: synced to s3://%s/%s", bucket, s3_key)
    except Exception:
        log.warning(
            "bakeoff: S3 sync FAILED for %s-%s — local copy at %s is still "
            "intact, but this run's evidence is NOT yet durable past a box "
            "replacement. Investigate (IAM grant on ops/bakeoff/*? "
            "network?).",
            date_str, edition, local_path, exc_info=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Shadow-canary bakeoff: prod (Anthropic) vs. Phase-B OSS "
            "candidates (OpenRouter) coverage-guard parity (config#1659). "
            "Runs a real (billed) grounded call per side; publishes none "
            "of them."
        )
    )
    parser.add_argument(
        "--date", default=None,
        help="YYYY-MM-DD (default: today, UTC-naive — matches the "
             "production episode's own date_str convention)",
    )
    parser.add_argument("--edition", default="am", choices=["am", "pm"])
    parser.add_argument(
        "--min-interval-days", type=int, default=DEFAULT_MIN_INTERVAL_DAYS,
        help="refuse a billed comparison this soon after the last completed "
             f"one (default: {DEFAULT_MIN_INTERVAL_DAYS}); 0 disables the guard",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="run even inside the minimum interval — for a deliberate operator "
             "re-run, never for a scheduled one",
    )
    args = parser.parse_args()

    date_str = args.date or _dt.date.today().isoformat()
    log_dir = Path(os.environ.get(BAKEOFF_LOG_DIR_ENV, DEFAULT_BAKEOFF_LOG_DIR))

    # BEFORE the AWS/SSM bootstrap and before any billed call: the cheapest
    # possible place to refuse. Exit 0, not 1 — an off-schedule LAUNCH is not a
    # bakeoff failure, and paging for it would manufacture exactly the red unit
    # this guard exists to prevent.
    last_run = None if args.force else _last_bakeoff_date(log_dir)
    if last_run is not None and args.min_interval_days > 0:
        gap_days = (_dt.date.fromisoformat(date_str) - last_run).days
        if gap_days < args.min_interval_days:
            log.warning(
                "bakeoff %s-%s: refusing — last completed comparison was %s "
                "(%d day(s) ago, minimum interval %d). This is a weekly, BILLED "
                "comparison; something started it off-schedule. Re-run "
                "deliberately with --force.",
                date_str, args.edition, last_run.isoformat(), gap_days,
                args.min_interval_days,
            )
            return 0

    try:
        # Mirrors episode.py/cli.py's bootstrap order exactly: assume the
        # runner role BEFORE touching SSM/S3. Missing this step silently
        # falls back to the box's own EC2 instance-profile credentials
        # (alpha-engine-dashboard-role), which have no S3 grant on
        # morning-signal-podcast at all — masked for months by that
        # bucket's public-read policy (fixed 2026-07-06, PR #104) until
        # this exact script's own verification run surfaced it as an
        # AccessDenied on prompts/prompt.md.
        _aws._AWS_SESSION = _aws._load_runner_session()
        _maybe_load_from_ssm()
    except Exception as exc:
        log.error(
            "bakeoff: SSM bootstrap failed (%s: %s) — the production "
            "service would fail the same way.",
            type(exc).__name__, exc,
        )
        return 1

    if not os.environ.get("OPENROUTER_API_KEY"):
        log.error(
            "bakeoff: OPENROUTER_API_KEY not set. Provision "
            "/morning-signal/openrouter-api-key in SSM (config#1659 gate) "
            "or export it locally for a one-off run."
        )
        return 1

    try:
        config = load_config()
    except Exception as exc:
        log.error(
            "bakeoff: load_config() failed (%s: %s)",
            type(exc).__name__, exc,
        )
        return 1

    try:
        record = run_bakeoff(config, date_str, args.edition)
    except Exception:
        log.exception("bakeoff: run failed for %s-%s", date_str, args.edition)
        return 1

    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"{date_str}-{args.edition}.bakeoff.jsonl"
    with out_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    _sync_to_s3(config, out_path, date_str, args.edition)

    any_worse = False
    for label, candidate in record["candidates"].items():
        log.info(
            "bakeoff %s-%s: prod unmet=%s %s unmet=%s parity=%s -> %s",
            date_str, args.edition,
            record["prod"]["unmet_topics"], label, candidate["unmet_topics"],
            candidate["parity"], out_path,
        )
        if candidate["parity"]["candidate_strictly_worse"]:
            any_worse = True
            log.warning(
                "bakeoff %s-%s: %s covered FEWER required topics than prod "
                "on the identical prompt — do not advance this candidate's "
                "flip until this stops recurring.",
                date_str, args.edition, label,
            )
    if not any_worse:
        log.info(
            "bakeoff %s-%s: no candidate was strictly worse than prod this "
            "run.",
            date_str, args.edition,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
