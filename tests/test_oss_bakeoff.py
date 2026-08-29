"""Tests for ``scripts/oss_bakeoff.py`` (config#1659 Phase B shadow canary).

Mirrors ``tests/test_canary.py``'s philosophy: stub the network boundary
(``LLMClient.complete_grounded``) and exercise the parity-comparison logic
and the exit-code matrix without hitting a real Anthropic or OpenRouter
call.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from krepis.llm import GroundedResult, LLMUsage
from krepis.llm_config import ModelSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


class _FakeS3:
    """No-op S3 client stand-in — records upload_file calls, never touches
    the network. The default for every test unless a test overrides
    bakeoff_module._aws_client itself to exercise sync success/failure."""

    def __init__(self):
        self.uploads = []

    def upload_file(self, local_path, bucket, key, **kw):
        self.uploads.append((local_path, bucket, key, kw))


@pytest.fixture
def bakeoff_module(monkeypatch, tmp_path: Path):
    """Reload ``oss_bakeoff`` with config/prompt/env wired to tmp paths —
    same shape as ``test_canary.py``'s ``canary_module`` fixture."""
    from morning_signal import config as _config_mod

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "s3_bucket: test-bucket\n"
        "claude_model: claude-haiku-4-5\n"
        "max_tokens: 4096\n"
        "web_search_max_uses: 20\n"
        "min_grounding_citations: 1\n"
        "required_search_topics:\n"
        "  - name: Political pulse\n"
        "    keywords: [maga]\n"
    )
    (tmp_path / "prompt.md").write_text("Weekday system prompt.\n")
    (tmp_path / "prompt_weekend.md").write_text("Weekend system prompt.\n")

    monkeypatch.setattr(_config_mod, "CONFIG_FILE", cfg_path)
    monkeypatch.setattr(_config_mod, "PROMPT_FILE", tmp_path / "prompt.md")
    monkeypatch.setattr(
        _config_mod, "PROMPT_WEEKEND_FILE", tmp_path / "prompt_weekend.md"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    monkeypatch.setenv("MORNING_SIGNAL_BAKEOFF_ALLOW_DIRECT_OPENROUTER", "1")
    monkeypatch.delenv("MORNING_SIGNAL_USE_SSM", raising=False)
    monkeypatch.chdir(tmp_path)

    if "oss_bakeoff" in sys.modules:
        del sys.modules["oss_bakeoff"]
    module = importlib.import_module("oss_bakeoff")
    # Never let a test hit real AWS for the S3 sync — default to a no-op
    # fake unless a test explicitly wants to inspect/break the sync.
    monkeypatch.setattr(module, "_aws_client", lambda *a, **kw: _FakeS3())
    return module


def _grounded(*, provider, model, unmet_hit, web_search_requests):
    """A GroundedResult whose script/citations satisfy the 'Political
    pulse' topic (keyword 'maga') iff ``unmet_hit`` is False."""
    text = "Welcome to Morning Signal. MAGA reacted today." if not unmet_hit else "Welcome to Morning Signal. Markets only."
    citations = [] if unmet_hit else [{"url": "https://x", "title": "maga rally", "snippet": None}]
    return GroundedResult(
        text=text,
        model=model,
        provider=provider,
        usage=LLMUsage(web_search_requests=web_search_requests),
        raw_request={},
        raw_response=None,
        searches=[],
        citations=citations,
    )


def _dispatcher(plan_by_provider_or_model):
    """A LLMClient fake dispatching by spec.provider (prod, "anthropic")
    or spec.model (each OpenRouter candidate has a distinct model id, so
    provider alone can't disambiguate between them)."""

    class _Client:
        def __init__(self, spec: ModelSpec, **kw):
            self.spec = spec

        def complete_grounded(self, **kw):
            key = self.spec.model if self.spec.provider == "openrouter" else self.spec.provider
            return plan_by_provider_or_model[key]

    return _Client


def _all_pass_plan(*, prod_unmet=False):
    return {
        "anthropic": _grounded(
            provider="anthropic", model="claude-haiku-4-5",
            unmet_hit=prod_unmet, web_search_requests=2,
        ),
        "moonshotai/kimi-k2.6": _grounded(
            provider="openrouter", model="moonshotai/kimi-k2.6",
            unmet_hit=False, web_search_requests=2,
        ),
        "xiaomi/mimo-v2.5-pro": _grounded(
            provider="openrouter", model="xiaomi/mimo-v2.5-pro",
            unmet_hit=False, web_search_requests=2,
        ),
    }


def test_prod_side_survives_a_spec_with_no_server_side_web_search(
    bakeoff_module, monkeypatch
):
    """Regression, live failure 2026-08-05 through 2026-08-08.

    When the production ``llm`` config moved to a krepis router group
    (provider ``litellm``, #135), this script's prod side still issued a bare
    ``complete_grounded`` — which that transport does not support — so the
    weekly unit exited 1 on every run while the production episode, which
    degrades to ``complete()``, aired normally. The bakeoff must exercise
    whatever spec production actually uses, or it measures nothing.
    """
    from krepis.llm import LLMResult
    from krepis.llm_config import LLMConfigError

    monkeypatch.setenv("MORNING_SIGNAL_LLM", '{"provider": "litellm", "model": "high"}')
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

    plan = _all_pass_plan()

    class _Client:
        def __init__(self, spec: ModelSpec, **kw):
            self.spec = spec

        def complete_grounded(self, **kw):
            if self.spec.provider == "litellm_proxy":
                raise LLMConfigError(
                    "complete_grounded is only supported on the anthropic "
                    "provider or openrouter; provider 'litellm_proxy' has "
                    "neither."
                )
            return plan[self.spec.model]

        def complete(self, **kw):
            assert kw["on_unsupported"] == "drop", (
                "the prod side must degrade the same way production does"
            )
            grounded = _grounded(
                provider="litellm_proxy", model="high",
                unmet_hit=False, web_search_requests=0,
            )
            return LLMResult(
                text=grounded.text,
                model=grounded.model,
                provider=grounded.provider,
                usage=grounded.usage,
                raw_request={},
                raw_response=None,
            )

    monkeypatch.setattr(bakeoff_module, "LLMClient", _Client)

    from morning_signal.config import load_config

    record = bakeoff_module.run_bakeoff(load_config(), "2026-07-06", "am")

    assert record["prod"]["provider"] == "litellm_proxy"
    assert record["prod"]["model"] == "high"
    # A no-web-search transport reports zero searches by construction; the
    # comparison still ran, which is the whole point of the regression.
    assert record["prod"]["n_searches"] == 0
    assert set(record["candidates"]) == {"kimi-k2.6", "mimo-v2.5-pro"}


def test_run_bakeoff_reports_parity_for_every_candidate(bakeoff_module, monkeypatch):
    from morning_signal.config import load_config

    monkeypatch.setattr(bakeoff_module, "LLMClient", _dispatcher(_all_pass_plan()))

    config = load_config()
    record = bakeoff_module.run_bakeoff(config, "2026-07-06", "am")

    assert record["prod"]["unmet_topics"] == []
    assert set(record["candidates"]) == {"kimi-k2.6", "mimo-v2.5-pro"}
    for label in ("kimi-k2.6", "mimo-v2.5-pro"):
        candidate = record["candidates"][label]
        assert candidate["unmet_topics"] == []
        assert candidate["parity"]["unmet_topics_match"] is True
        assert candidate["parity"]["candidate_strictly_worse"] is False
        assert candidate["parity"]["both_met_grounding"] is True


def test_run_bakeoff_flags_one_candidate_strictly_worse(bakeoff_module, monkeypatch):
    plan = _all_pass_plan()
    plan["moonshotai/kimi-k2.6"] = _grounded(
        provider="openrouter", model="moonshotai/kimi-k2.6",
        unmet_hit=True, web_search_requests=2,
    )
    monkeypatch.setattr(bakeoff_module, "LLMClient", _dispatcher(plan))

    from morning_signal.config import load_config
    config = load_config()
    record = bakeoff_module.run_bakeoff(config, "2026-07-06", "am")

    kimi = record["candidates"]["kimi-k2.6"]
    mimo = record["candidates"]["mimo-v2.5-pro"]
    assert kimi["unmet_topics"] == ["Political pulse"]
    assert kimi["parity"]["candidate_strictly_worse"] is True
    assert mimo["unmet_topics"] == []
    assert mimo["parity"]["candidate_strictly_worse"] is False


def test_candidate_specs_carry_reasoning_exclude(bakeoff_module, monkeypatch):
    """Both candidates must set reasoning={"exclude": True} — the fix for
    the empty-content bug found 2026-07-06 (krepis#16)."""
    seen_specs = []

    class _Client:
        def __init__(self, spec, **kw):
            seen_specs.append(spec)
            self.spec = spec

        def complete_grounded(self, **kw):
            plan = _all_pass_plan()
            key = self.spec.model if self.spec.provider == "openrouter" else self.spec.provider
            return plan[key]

    monkeypatch.setattr(bakeoff_module, "LLMClient", _Client)

    from morning_signal.config import load_config
    config = load_config()
    bakeoff_module.run_bakeoff(config, "2026-07-06", "am")

    candidate_specs = [s for s in seen_specs if s.provider == "openrouter"]
    assert len(candidate_specs) == 2
    for spec in candidate_specs:
        assert spec.reasoning == {"exclude": True}


def test_main_assumes_runner_role_before_ssm_bootstrap(bakeoff_module, monkeypatch):
    """2026-07-06 regression: main() used to call _maybe_load_from_ssm()
    without first assuming the runner role — see the identical fix +
    rationale in tests/test_canary.py."""
    from morning_signal import aws as _aws_mod

    sentinel = object()
    observed = []

    monkeypatch.setattr(_aws_mod, "_load_runner_session", lambda: sentinel)
    monkeypatch.setattr(_aws_mod, "_AWS_SESSION", None)

    def fake_maybe_load_from_ssm():
        observed.append(_aws_mod._AWS_SESSION)
        raise RuntimeError("stop here — order already observed")

    monkeypatch.setattr(bakeoff_module, "_maybe_load_from_ssm", fake_maybe_load_from_ssm)
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-06"])

    result = bakeoff_module.main()

    assert result == 1
    assert observed == [sentinel]


def test_main_fails_without_openrouter_key(bakeoff_module, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-06"])
    assert bakeoff_module.main() == 1


# ── Direct-OpenRouter compliance gate (morning-signal-I165, 2026-08-29) ─────


def test_main_refuses_direct_openrouter_dispatch_by_default(
    bakeoff_module, monkeypatch
):
    """Without the explicit override, main() must refuse BEFORE the AWS/SSM
    bootstrap and before any billed call — CANDIDATES address OpenRouter
    directly (bare ModelSpec), which alpha-engine-config-I6367 and Brian's
    2026-08-29 "no parallel setups, funnel through the krepis router" ruling
    both forbid. The shipped systemd unit never sets the override, so this
    is what actually governs the weekly scheduled run."""
    monkeypatch.delenv("MORNING_SIGNAL_BAKEOFF_ALLOW_DIRECT_OPENROUTER", raising=False)

    def _unreachable_ssm():
        raise AssertionError(
            "main() must refuse the direct-OpenRouter dispatch before ever "
            "touching AWS/SSM, not merely before the LLM call"
        )

    monkeypatch.setattr(bakeoff_module, "_maybe_load_from_ssm", _unreachable_ssm)
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-06"])

    assert bakeoff_module.main() == 1


def test_main_allows_direct_openrouter_dispatch_with_explicit_override(
    bakeoff_module, monkeypatch, tmp_path
):
    """The override is the deliberate, human-aware escape hatch for a
    manual one-off — with it set, the comparison runs exactly as before."""
    monkeypatch.setenv("MORNING_SIGNAL_BAKEOFF_ALLOW_DIRECT_OPENROUTER", "1")
    monkeypatch.setattr(bakeoff_module, "LLMClient", _dispatcher(_all_pass_plan()))
    log_dir = tmp_path / "bakeoff_out"
    monkeypatch.setenv(bakeoff_module.BAKEOFF_LOG_DIR_ENV, str(log_dir))
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-06", "--edition", "am"])

    assert bakeoff_module.main() == 0
    assert (log_dir / "2026-07-06-am.bakeoff.jsonl").exists()


def test_main_writes_jsonl_on_success(bakeoff_module, monkeypatch, tmp_path):
    monkeypatch.setattr(bakeoff_module, "LLMClient", _dispatcher(_all_pass_plan()))
    log_dir = tmp_path / "bakeoff_out"
    monkeypatch.setenv(bakeoff_module.BAKEOFF_LOG_DIR_ENV, str(log_dir))
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-06", "--edition", "am"])

    assert bakeoff_module.main() == 0

    out_path = log_dir / "2026-07-06-am.bakeoff.jsonl"
    assert out_path.exists()
    record = json.loads(out_path.read_text().strip().splitlines()[0])
    assert record["date"] == "2026-07-06"
    assert record["edition"] == "am"
    assert record["candidates"]["kimi-k2.6"]["parity"]["unmet_topics_match"] is True
    assert record["candidates"]["mimo-v2.5-pro"]["parity"]["unmet_topics_match"] is True


def test_main_appends_on_repeated_runs(bakeoff_module, monkeypatch, tmp_path):
    """--force is what a DELIBERATE repeat looks like now: the second run is
    inside the minimum interval, so an unforced one would (correctly) refuse.
    What this test still pins is the append, not the cadence."""
    monkeypatch.setattr(bakeoff_module, "LLMClient", _dispatcher(_all_pass_plan()))
    log_dir = tmp_path / "bakeoff_out"
    monkeypatch.setenv(bakeoff_module.BAKEOFF_LOG_DIR_ENV, str(log_dir))
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-06", "--edition", "am"])
    assert bakeoff_module.main() == 0

    monkeypatch.setattr(
        sys, "argv",
        ["oss_bakeoff.py", "--date", "2026-07-06", "--edition", "am", "--force"],
    )
    assert bakeoff_module.main() == 0

    out_path = log_dir / "2026-07-06-am.bakeoff.jsonl"
    assert len(out_path.read_text().strip().splitlines()) == 2


# ── Minimum-interval guard (alpha-engine-config-I9000) ──────────────────────
#
# 2026-08-28 03:01 UTC: this script ran on a Friday because an unrelated
# crucible-dashboard deploy re-ran the systemd installer and `Requires=` in the
# timer's [Unit] made `enable --now <timer>` start the service as a dependency.
# The launcher is fixed there; these pin the defence in depth here, because a
# workload that spends money per run must not depend on its launcher being
# right — and self-hosters supply their own launcher entirely.
#
# Every test below asserts BEHAVIOUR — whether a billed comparison happened,
# read off the JSONL the run writes — not the presence of a flag or a string.


def _bill_counting_dispatcher(counter):
    """A dispatcher that records every LLM call, so 'did this run cost money?'
    is an assertion rather than an inference from the exit code."""
    inner = _dispatcher(_all_pass_plan())

    class _Counting(inner):
        def complete_grounded(self, **kw):
            counter.append(self.spec.model)
            return super().complete_grounded(**kw)

    return _Counting


def test_a_run_inside_the_minimum_interval_makes_no_billed_call(
    bakeoff_module, monkeypatch, tmp_path
):
    calls: list = []
    monkeypatch.setattr(bakeoff_module, "LLMClient", _bill_counting_dispatcher(calls))
    log_dir = tmp_path / "bakeoff_out"
    monkeypatch.setenv(bakeoff_module.BAKEOFF_LOG_DIR_ENV, str(log_dir))

    # A completed comparison two days earlier — the real shape of the incident:
    # last legitimate run Wednesday, off-schedule launch Friday.
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-01"])
    assert bakeoff_module.main() == 0
    assert calls, "setup: the first run should have made billed calls"
    calls.clear()

    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-03"])
    # Exit 0, not 1: an off-schedule LAUNCH is not a bakeoff failure, and
    # exiting non-zero would manufacture the failed unit + page this whole fix
    # exists to remove.
    assert bakeoff_module.main() == 0
    assert calls == [], "a refused run must not reach the model"
    assert not (log_dir / "2026-07-03-am.bakeoff.jsonl").exists()


def test_the_next_weekly_run_still_happens(bakeoff_module, monkeypatch, tmp_path):
    """The half that must not be traded away for the half above."""
    calls: list = []
    monkeypatch.setattr(bakeoff_module, "LLMClient", _bill_counting_dispatcher(calls))
    log_dir = tmp_path / "bakeoff_out"
    monkeypatch.setenv(bakeoff_module.BAKEOFF_LOG_DIR_ENV, str(log_dir))

    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-01"])
    assert bakeoff_module.main() == 0
    calls.clear()

    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-08"])
    assert bakeoff_module.main() == 0
    assert calls, "the next Wednesday is due and must run"
    assert (log_dir / "2026-07-08-am.bakeoff.jsonl").exists()


def test_a_persistent_catch_up_a_few_days_late_still_runs(
    bakeoff_module, monkeypatch, tmp_path
):
    """`Persistent=true` replays a slot missed while the box was down, which
    lands the weekly run DAYS late. A day-of-week window would refuse it; a
    minimum interval must not. This is why the guard is an interval."""
    calls: list = []
    monkeypatch.setattr(bakeoff_module, "LLMClient", _bill_counting_dispatcher(calls))
    log_dir = tmp_path / "bakeoff_out"
    monkeypatch.setenv(bakeoff_module.BAKEOFF_LOG_DIR_ENV, str(log_dir))

    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-01"])
    assert bakeoff_module.main() == 0
    calls.clear()

    # Box down over the 07-08 slot; comes back 07-11 and systemd replays it.
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-11"])
    assert bakeoff_module.main() == 0
    assert calls, "a late catch-up is still a due run"


def test_force_overrides_the_interval(bakeoff_module, monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(bakeoff_module, "LLMClient", _bill_counting_dispatcher(calls))
    log_dir = tmp_path / "bakeoff_out"
    monkeypatch.setenv(bakeoff_module.BAKEOFF_LOG_DIR_ENV, str(log_dir))

    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-01"])
    assert bakeoff_module.main() == 0
    calls.clear()

    monkeypatch.setattr(
        sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-02", "--force"]
    )
    assert bakeoff_module.main() == 0
    assert calls, "--force is the deliberate operator re-run"


def test_a_failed_run_leaves_no_marker_so_its_retry_is_not_blocked(
    bakeoff_module, monkeypatch, tmp_path
):
    """The marker is the completed-comparison log. A setup failure writes none,
    so the guard must never turn one bad run into a week of silence."""
    log_dir = tmp_path / "bakeoff_out"
    monkeypatch.setenv(bakeoff_module.BAKEOFF_LOG_DIR_ENV, str(log_dir))

    real_run_bakeoff = bakeoff_module.run_bakeoff

    def _boom(*a, **kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr(bakeoff_module, "run_bakeoff", _boom)
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-01"])
    assert bakeoff_module.main() == 1
    assert not list(log_dir.glob("*.bakeoff.jsonl")) if log_dir.exists() else True

    monkeypatch.setattr(bakeoff_module, "run_bakeoff", real_run_bakeoff)
    calls: list = []
    monkeypatch.setattr(bakeoff_module, "LLMClient", _bill_counting_dispatcher(calls))
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-02"])
    assert bakeoff_module.main() == 0
    assert calls, "the retry after a failure must not be blocked by the guard"


def test_the_first_ever_run_is_not_blocked(bakeoff_module, monkeypatch, tmp_path):
    """An empty (or absent) log dir must read as 'never run', not as 'today'."""
    calls: list = []
    monkeypatch.setattr(bakeoff_module, "LLMClient", _bill_counting_dispatcher(calls))
    monkeypatch.setenv(
        bakeoff_module.BAKEOFF_LOG_DIR_ENV, str(tmp_path / "never-created")
    )
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-01"])

    assert bakeoff_module.main() == 0
    assert calls


# ── S3 sync (durability across box replacement, 2026-07-06) ─────────────────


def test_main_syncs_to_s3_ops_bakeoff_prefix(bakeoff_module, monkeypatch, tmp_path):
    monkeypatch.setattr(bakeoff_module, "LLMClient", _dispatcher(_all_pass_plan()))
    fake_s3 = _FakeS3()
    monkeypatch.setattr(bakeoff_module, "_aws_client", lambda *a, **kw: fake_s3)
    log_dir = tmp_path / "bakeoff_out"
    monkeypatch.setenv(bakeoff_module.BAKEOFF_LOG_DIR_ENV, str(log_dir))
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-06", "--edition", "am"])

    assert bakeoff_module.main() == 0

    assert len(fake_s3.uploads) == 1
    local_path, bucket, key, kw = fake_s3.uploads[0]
    assert bucket == "test-bucket"
    assert key == "ops/bakeoff/2026-07-06-am.bakeoff.jsonl"
    assert str(local_path).endswith("2026-07-06-am.bakeoff.jsonl")


def test_sync_failure_does_not_crash_the_run(bakeoff_module, monkeypatch, tmp_path):
    """S3 sync is secondary to the local write — a failure there must not
    take down the whole bakeoff run."""
    class _BrokenS3:
        def upload_file(self, *a, **kw):
            raise RuntimeError("simulated S3 outage")

    monkeypatch.setattr(bakeoff_module, "LLMClient", _dispatcher(_all_pass_plan()))
    monkeypatch.setattr(bakeoff_module, "_aws_client", lambda *a, **kw: _BrokenS3())
    log_dir = tmp_path / "bakeoff_out"
    monkeypatch.setenv(bakeoff_module.BAKEOFF_LOG_DIR_ENV, str(log_dir))
    monkeypatch.setattr(sys, "argv", ["oss_bakeoff.py", "--date", "2026-07-06", "--edition", "am"])

    assert bakeoff_module.main() == 0

    out_path = log_dir / "2026-07-06-am.bakeoff.jsonl"
    assert out_path.exists()  # local copy still intact despite the S3 failure


def test_sync_skipped_with_warning_when_no_s3_bucket_configured(
    bakeoff_module, monkeypatch, tmp_path, caplog
):
    import logging

    fake_s3 = _FakeS3()
    monkeypatch.setattr(bakeoff_module, "_aws_client", lambda *a, **kw: fake_s3)
    from morning_signal.config import load_config
    config = load_config()
    config.pop("s3_bucket", None)

    log_dir = tmp_path / "bakeoff_out"
    out_path = log_dir / "2026-07-06-am.bakeoff.jsonl"
    log_dir.mkdir(parents=True)
    out_path.write_text("{}\n")

    with caplog.at_level(logging.WARNING):
        bakeoff_module._sync_to_s3(config, out_path, "2026-07-06", "am")

    assert fake_s3.uploads == []
    assert any("no s3_bucket" in r.message for r in caplog.records)
