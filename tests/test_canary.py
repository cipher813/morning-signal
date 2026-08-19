"""Tests for ``scripts/canary.py`` (ROADMAP L380 Phase A).

The script dispatches a live ``max_tokens=1`` call through
``krepis.llm.LLMClient`` in production — routed through the krepis router
as of alpha-engine-config-I6980, rather than a direct
``anthropic.Anthropic`` client. These tests stub the network boundary
(patching ``sys.modules["anthropic"]``, which krepis's own lazy
``import anthropic`` picks up transparently for the anthropic-transport
path the OSS legacy-default fixture config resolves to) and exercise the
exit-code matrix (no API key / SSM failure / config load failure / request-
build failure / HTTP 400 / HTTP 5xx / OK).

Mirrors ``tests/live_api_smoke.py``'s philosophy: validate the
producer-side surface (config + prompt + spec-resolution chain) without
hitting the real API. Live API coverage is the CI smoke
(``.github/workflows/live-api-smoke.yml``) plus production runtime.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture
def canary_module(monkeypatch, tmp_path: Path):
    """Reload ``canary`` after wiring config + prompt + env to tmp paths.

    The fixture writes a minimal ``config.yaml`` + ``prompt.md`` /
    ``prompt_weekend.md`` pair, points ``morning_signal.config``
    module-level paths at them, sets ``ANTHROPIC_API_KEY``, and unsets
    ``MORNING_SIGNAL_USE_SSM`` so the SSM bootstrap is a no-op (the
    live-SSM path is exercised in ``tests/test_aws_paths.py``).
    """
    from morning_signal import config as _config_mod

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "s3_bucket: test-bucket\n"
        "claude_model: claude-sonnet-4-6\n"
        "max_tokens: 4096\n"
        "web_search_max_uses: 20\n"
    )
    (tmp_path / "prompt.md").write_text("Weekday system prompt.\n")
    (tmp_path / "prompt_weekend.md").write_text("Weekend system prompt.\n")

    monkeypatch.setattr(_config_mod, "CONFIG_FILE", cfg_path)
    monkeypatch.setattr(_config_mod, "PROMPT_FILE", tmp_path / "prompt.md")
    monkeypatch.setattr(
        _config_mod, "PROMPT_WEEKEND_FILE", tmp_path / "prompt_weekend.md"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.delenv("MORNING_SIGNAL_USE_SSM", raising=False)

    if "canary" in sys.modules:
        del sys.modules["canary"]
    return importlib.import_module("canary")


def test_canary_returns_1_when_api_key_missing(canary_module, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert canary_module.main() == 1


def test_canary_assumes_runner_role_before_ssm_bootstrap(canary_module, monkeypatch):
    """2026-07-06 regression: main() used to call _maybe_load_from_ssm()
    without first assuming the runner role, silently falling back to the
    box's own EC2 instance-profile credentials — masked for months by
    morning-signal-podcast's public-read bucket policy until that policy
    was scoped down (PR #104) and this exact gap surfaced as a live
    AccessDenied. episode.py/cli.py always assumed the role first; the
    scripts must match that order.

    Observes the session state INSIDE the faked SSM bootstrap, then raises
    to short-circuit before any real API dispatch — canary's own
    try/except turns that into a clean ``1``, so the assertion on what was
    observed happens outside that except block, not inside it (an
    AssertionError raised there would otherwise be silently swallowed and
    misreported as an ordinary bootstrap failure).
    """
    from morning_signal import aws as _aws_mod

    sentinel = object()
    observed = []

    monkeypatch.setattr(_aws_mod, "_load_runner_session", lambda: sentinel)
    monkeypatch.setattr(_aws_mod, "_AWS_SESSION", None)

    def fake_maybe_load_from_ssm():
        observed.append(_aws_mod._AWS_SESSION)
        raise RuntimeError("stop here — order already observed")

    monkeypatch.setattr(canary_module, "_maybe_load_from_ssm", fake_maybe_load_from_ssm)

    result = canary_module.main()

    assert result == 1
    assert observed == [sentinel]


def test_canary_returns_1_on_invalid_edition(canary_module, monkeypatch):
    monkeypatch.setenv("MORNING_SIGNAL_CANARY_EDITION", "midnight")
    assert canary_module.main() == 1


def test_canary_builds_production_shape_request(canary_module):
    """The canary's ``(system, user_content)`` pair MUST mirror
    ``generate_script``'s ``build_episode_request`` shape — the actual
    payload (server-tool, cache_control, max_tokens, no assistant prefill)
    is now built and validated by ``krepis.llm.LLMClient`` at dispatch time,
    the SAME code path ``generate_script`` calls through (see
    ``test_canary_returns_0_on_successful_dispatch`` for that path's
    coverage).
    """
    prompt_text, user_content = canary_module._build_canary_request("2026-05-28", "am")

    assert prompt_text == "Weekday system prompt."
    assert "2026-05-28" not in prompt_text  # date lives in user_content, not system
    assert "Thursday, May 28, 2026" in user_content
    assert "MORNING edition" in user_content
    assert "MUST begin verbatim" in user_content


def test_canary_resolves_the_same_spec_generate_script_would(canary_module):
    """The canary resolves through ``resolve_llm_spec`` — the SAME function
    ``generate_script``'s primary calls — so a router group that will not
    resolve here fails the canary identically to how it would fail the
    real episode. The fixture config has no ``llm`` key, so this locks in
    the legacy anthropic-default resolution path (OSS self-host mode,
    outside I6980's scope) rather than a router group.
    """
    from morning_signal.claude import resolve_llm_spec

    cfg = {"claude_model": "claude-sonnet-4-6", "max_tokens": 4096}
    spec = resolve_llm_spec(cfg)
    assert spec.provider == "anthropic"
    assert spec.model == "claude-sonnet-4-6"


def test_canary_returns_0_on_successful_dispatch(canary_module):
    fake_resp = MagicMock()
    fake_resp.stop_reason = "max_tokens"
    fake_resp.usage.input_tokens = 1234
    fake_resp.usage.output_tokens = 1

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_resp

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    fake_anthropic.BadRequestError = type("BadRequestError", (Exception,), {})
    fake_anthropic.APIStatusError = type("APIStatusError", (Exception,), {})

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        assert canary_module.main() == 0
    fake_client.messages.create.assert_called_once()


def test_canary_returns_1_on_anthropic_400(canary_module):
    class _BadRequest(Exception):
        pass

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = _BadRequest(
        "This model does not support assistant message prefill."
    )

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    fake_anthropic.BadRequestError = _BadRequest
    fake_anthropic.APIStatusError = type("APIStatusError", (Exception,), {})

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        assert canary_module.main() == 1


# ── SSM-sourced key (2026-07-06 bootstrap-order bug) ─────────────────────────
#
# Production (MORNING_SIGNAL_USE_SSM=1) never sets ANTHROPIC_API_KEY via the
# systemd unit's Environment= directives — _maybe_load_from_ssm() is what
# populates it, from /morning-signal/anthropic-api-key. The canary used to
# check for the env var BEFORE calling that bootstrap, so it could never
# actually pass when run the way the real service is invoked; only a local
# run with the key pre-exported (bypassing SSM entirely) ever exercised the
# success path. This locks down the fix: SSM is the ONLY source of the key
# here (no local ANTHROPIC_API_KEY pre-set), matching the live production
# shape.


@mock_aws
def test_canary_succeeds_when_key_comes_only_from_ssm(monkeypatch):
    region = "us-east-1"
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", region)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MORNING_SIGNAL_CANARY_EDITION", raising=False)

    ssm = boto3.client("ssm", region_name=region)
    ssm.put_parameter(
        Name="/morning-signal/anthropic-api-key",
        Value="sk-from-ssm-only", Type="SecureString",
    )
    ssm.put_parameter(
        Name="/morning-signal/config-yaml",
        Value=(
            "s3_bucket: test-bucket\n"
            "claude_model: claude-sonnet-4-6\n"
            "max_tokens: 4096\n"
            "web_search_max_uses: 20\n"
            # `llm` is REQUIRED on a MORNING_SIGNAL_USE_SSM=1 deployment as of
            # this branch: the call site fails closed rather than defaulting to
            # a hardcoded Anthropic model (model-router-policy). This test's
            # subject is that the API key resolves from SSM and not from the
            # process environment — the missing key made its FIXTURE incomplete
            # under the new contract; it did not make the assertion wrong.
            #
            # Deliberately a direct-Anthropic spec, NOT a router group: this
            # test's seam is a mocked ``anthropic`` module and it asserts
            # ``fake_client.messages.create`` was actually called — a router
            # group would send the call through ``_resolve_router_group`` /
            # ``krepis.router.resolve_group_spec`` instead, which needs a real
            # registry (LLM_MODEL_REGISTRY.yaml) or an authenticated LiteLLM
            # edge this test never mocks. Router-group resolution has its own
            # coverage elsewhere; this fixture only needs `llm` to be SET so
            # the SSM-mode fail-closed guard does not fire.
            #
            # The value is a JSON STRING, not a nested YAML mapping.
            # `declared_llm_spec` does `raw = str(configured)` then parses raw
            # as JSON, so a YAML mapping arrives as a single-quoted Python repr
            # and fails with "Expecting property name enclosed in double
            # quotes". Matches the form the error message itself prescribes.
            "llm: '{\"provider\": \"anthropic\", \"model\": \"claude-sonnet-4-6\"}'\n"
        ),
        Type="SecureString",
    )

    s3 = boto3.client("s3", region_name="us-west-2")
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
    )
    s3.put_object(Bucket="test-bucket", Key="prompts/prompt.md", Body=b"Weekday prompt.")

    if "canary" in sys.modules:
        del sys.modules["canary"]
    canary_module = importlib.import_module("canary")

    fake_resp = MagicMock()
    fake_resp.stop_reason = "max_tokens"
    fake_resp.usage.input_tokens = 1234
    fake_resp.usage.output_tokens = 1

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_resp

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    fake_anthropic.BadRequestError = type("BadRequestError", (Exception,), {})
    fake_anthropic.APIStatusError = type("APIStatusError", (Exception,), {})

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        assert canary_module.main() == 0

    # The key that reached the Anthropic client came from SSM, not env.
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-from-ssm-only"
    fake_client.messages.create.assert_called_once()


def test_canary_returns_1_on_anthropic_5xx(canary_module):
    class _APIStatusError(Exception):
        def __init__(self, status_code: int, message: str) -> None:
            super().__init__(message)
            self.status_code = status_code

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = _APIStatusError(
        503, "service unavailable"
    )

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    fake_anthropic.BadRequestError = type("BadRequestError", (Exception,), {})
    fake_anthropic.APIStatusError = _APIStatusError

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        assert canary_module.main() == 1
