"""Tests for AWS-dependent paths via moto mocks (no real AWS calls)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
import pytest
from moto import mock_aws


REGION_S3 = "us-west-2"
REGION_OTHER = "us-east-1"


@pytest.fixture
def aws_env(monkeypatch):
    """Set fake AWS credentials so boto3 + moto are happy."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION_S3)


# ── _aws_client routing ──────────────────────────────────────────────────────


def test_aws_client_uses_default_chain_when_session_none(fresh_ge_module, aws_env):
    fresh_ge_module._AWS_SESSION = None
    c = fresh_ge_module._aws_client("s3", region_name=REGION_S3)
    # boto3.client returns a botocore client; just confirm we got *something* boto-shaped.
    assert hasattr(c, "list_buckets")


def test_aws_client_routes_through_session_when_set(fresh_ge_module, aws_env):
    sess = boto3.Session(region_name=REGION_S3)
    fresh_ge_module._AWS_SESSION = sess
    c = fresh_ge_module._aws_client("s3")
    assert hasattr(c, "list_buckets")
    # Confirm region propagation
    assert c.meta.region_name == REGION_S3


# ── _load_runner_session ─────────────────────────────────────────────────────


@mock_aws
def test_load_runner_session_returns_none_without_env_var(fresh_ge_module, monkeypatch):
    monkeypatch.delenv("MORNING_SIGNAL_RUNNER_ROLE_ARN", raising=False)
    assert fresh_ge_module._load_runner_session() is None


@mock_aws
def test_load_runner_session_assumes_role_when_env_set(
    fresh_ge_module, aws_env, monkeypatch
):
    iam = boto3.client("iam", region_name="us-east-1")
    iam.create_role(
        RoleName="test-runner",
        AssumeRolePolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": "*"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
        ),
    )
    role_arn = "arn:aws:iam::123456789012:role/test-runner"
    monkeypatch.setenv("MORNING_SIGNAL_RUNNER_ROLE_ARN", role_arn)

    session = fresh_ge_module._load_runner_session()
    assert session is not None
    # The returned session should have credentials present (moto provides fake ones).
    assert session.get_credentials() is not None


# ── _maybe_load_from_ssm ─────────────────────────────────────────────────────


@mock_aws
def test_maybe_load_from_ssm_is_noop_without_flag(fresh_ge_module, monkeypatch):
    monkeypatch.delenv("MORNING_SIGNAL_USE_SSM", raising=False)
    before_config = fresh_ge_module.CONFIG_FILE
    before_prompt = fresh_ge_module.PROMPT_FILE
    fresh_ge_module._maybe_load_from_ssm()
    assert fresh_ge_module.CONFIG_FILE == before_config
    assert fresh_ge_module.PROMPT_FILE == before_prompt


def _seed_ssm_and_s3(
    *,
    region: str = REGION_OTHER,
    bucket: str = "test-bucket",
    prompts_prefix: str = "prompts/",
    config_extras: str = "",
    anthropic_key: str | None = "sk-fake-key",
    prompt_md: str = "# Test prompt",
    prompt_weekend_md: str | None = None,
    extra_ssm_params: dict[str, str] | None = None,
) -> None:
    """Seed the SSM params + S3 objects the bootstrap path reads.

    SSM keeps: anthropic-api-key (OPTIONAL — pass ``anthropic_key=None`` to
    omit it, e.g. an install running entirely on the krepis router) +
    config-yaml (small structured/secret). S3 keeps: prompt.md (+ optional
    weekend) at s3://{bucket}/{prompts_prefix}<file>.

    Bucket creation routed to ``REGION_S3`` (us-west-2) to match the
    aws_env fixture's ``AWS_DEFAULT_REGION``; moto rejects a
    LocationConstraint mismatch.
    """
    ssm = boto3.client("ssm", region_name=region)
    if anthropic_key is not None:
        ssm.put_parameter(
            Name="/morning-signal/anthropic-api-key", Value=anthropic_key, Type="SecureString",
        )
    config_body = f"s3_bucket: {bucket}\nprompts_s3_prefix: {prompts_prefix}\n"
    if config_extras:
        config_body += config_extras
    ssm.put_parameter(
        Name="/morning-signal/config-yaml", Value=config_body, Type="SecureString",
    )
    for name, value in (extra_ssm_params or {}).items():
        ssm.put_parameter(Name=name, Value=value, Type="SecureString")

    s3 = boto3.client("s3", region_name=REGION_S3)
    s3.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={"LocationConstraint": REGION_S3},
    )
    s3.put_object(Bucket=bucket, Key=f"{prompts_prefix}prompt.md", Body=prompt_md.encode())
    if prompt_weekend_md is not None:
        s3.put_object(
            Bucket=bucket, Key=f"{prompts_prefix}prompt_weekend.md",
            Body=prompt_weekend_md.encode(),
        )


@mock_aws
def test_maybe_load_from_ssm_fetches_and_overrides_paths(
    fresh_ge_module, aws_env, monkeypatch
):
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    _seed_ssm_and_s3(prompt_md="# Test prompt")

    fresh_ge_module._maybe_load_from_ssm()

    # Paths should have been redirected to a tmpdir
    assert "morning-signal-" in str(fresh_ge_module.CONFIG_FILE)
    assert "morning-signal-" in str(fresh_ge_module.PROMPT_FILE)
    # config-yaml from SSM, prompt.md from S3
    assert "s3_bucket: test-bucket" in fresh_ge_module.CONFIG_FILE.read_text()
    assert fresh_ge_module.PROMPT_FILE.read_text() == "# Test prompt"
    # ANTHROPIC_API_KEY should have been exported into env
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-fake-key"


@mock_aws
def test_maybe_load_from_ssm_tolerates_missing_anthropic_key(
    fresh_ge_module, aws_env, monkeypatch
):
    """2026-08-29: the Anthropic key moved from a REQUIRED `fetch()` to an
    optional one — Brian's ruling sets the direct-Anthropic API budget to
    $0, and production generation is meant to be served by the krepis
    router (``llm`` config key), which holds its own credentials. An
    install with no ``/morning-signal/anthropic-api-key`` SSM parameter
    (router-only, or the parameter deliberately removed once the router
    path is confirmed live) must still boot rather than hard-failing on a
    credential production doesn't need."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    _seed_ssm_and_s3(anthropic_key=None)

    fresh_ge_module._maybe_load_from_ssm()

    assert "ANTHROPIC_API_KEY" not in os.environ


@mock_aws
def test_maybe_load_from_ssm_anthropic_local_override_wins(
    fresh_ge_module, aws_env, monkeypatch
):
    """A pre-set ANTHROPIC_API_KEY (local debugging, or an operator-supplied
    self-host key) is not overwritten by SSM."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "local-override-key")

    _seed_ssm_and_s3(anthropic_key="ssm-key")

    fresh_ge_module._maybe_load_from_ssm()

    assert os.environ["ANTHROPIC_API_KEY"] == "local-override-key"


@mock_aws
def test_maybe_load_from_ssm_materializes_gcp_key(fresh_ge_module, aws_env, monkeypatch):
    """The optional GCP TTS key (SSM SecureString) is written to a tmpdir file
    and GOOGLE_APPLICATION_CREDENTIALS points at it (the SDK wants a path)."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    key_json = '{"type": "service_account", "project_id": "morning-signal"}'
    _seed_ssm_and_s3(extra_ssm_params={"/morning-signal/gcp-tts-key": key_json})

    fresh_ge_module._maybe_load_from_ssm()

    path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    assert "morning-signal-" in path
    assert Path(path).read_text() == key_json


@mock_aws
def test_maybe_load_from_ssm_no_gcp_key_leaves_env_unset(fresh_ge_module, aws_env, monkeypatch):
    """Polly installs (no gcp-tts-key param) must not set the env var or crash."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    _seed_ssm_and_s3()  # no gcp key seeded

    fresh_ge_module._maybe_load_from_ssm()

    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


@mock_aws
def test_maybe_load_from_ssm_local_gcp_override_wins(fresh_ge_module, aws_env, monkeypatch):
    """A pre-set GOOGLE_APPLICATION_CREDENTIALS (local run) is not overwritten."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/local/key.json")
    _seed_ssm_and_s3(extra_ssm_params={"/morning-signal/gcp-tts-key": '{"x": 1}'})

    fresh_ge_module._maybe_load_from_ssm()

    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "/local/key.json"


@mock_aws
def test_maybe_load_from_ssm_fails_loudly_when_s3_bucket_missing_from_config(
    fresh_ge_module, aws_env, monkeypatch
):
    """If config-yaml in SSM doesn't declare ``s3_bucket``, the bootstrap
    cannot locate the prompt objects. Per ``feedback_no_silent_fails``,
    raise loudly rather than silently leaving the personal prompt unset."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    ssm = boto3.client("ssm", region_name=REGION_OTHER)
    ssm.put_parameter(
        Name="/morning-signal/anthropic-api-key", Value="sk", Type="SecureString",
    )
    # config-yaml present but missing the s3_bucket key
    ssm.put_parameter(
        Name="/morning-signal/config-yaml", Value="claude_model: x\n",
        Type="SecureString",
    )

    with pytest.raises(RuntimeError, match="s3_bucket"):
        fresh_ge_module._maybe_load_from_ssm()


@mock_aws
def test_maybe_load_from_ssm_loads_optional_weekend_prompt(
    fresh_ge_module, aws_env, monkeypatch
):
    """When the weekend prompt exists in S3, it's staged to tmpdir and
    the config module's PROMPT_*_FILE paths point at the staged copies."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    _seed_ssm_and_s3(
        prompt_md="weekday",
        prompt_weekend_md="weekend deep dive",
    )

    fresh_ge_module._maybe_load_from_ssm()

    assert fresh_ge_module.PROMPT_FILE.read_text() == "weekday"
    assert fresh_ge_module.PROMPT_WEEKEND_FILE.read_text() == "weekend deep dive"


@mock_aws
def test_maybe_load_from_ssm_weekend_optional_falls_back_when_absent_from_s3(
    fresh_ge_module, aws_env, monkeypatch, caplog
):
    """During the migration rollout window the weekend object may not
    have been pushed to S3 yet. Boot should not hard-fail — fall back
    to the weekday prompt with a WARN log so non-trading-day editions
    still produce output until ``./sync.sh`` is run."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    _seed_ssm_and_s3(prompt_md="weekday only")  # no weekend prompt

    import logging
    with caplog.at_level(logging.WARNING):
        fresh_ge_module._maybe_load_from_ssm()

    # Fallback: weekend path equals weekday path
    assert fresh_ge_module.PROMPT_WEEKEND_FILE == fresh_ge_module.PROMPT_FILE
    # WARN was emitted
    assert any("prompt_weekend.md" in r.message for r in caplog.records)


@mock_aws
def test_maybe_load_from_ssm_loads_telegram_creds_when_present(
    fresh_ge_module, aws_env, monkeypatch
):
    """Optional flow-doctor / Telegram params under /morning-signal/*
    flow into env vars when present. Local env-var overrides win."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FLOW_DOCTOR_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("FLOW_DOCTOR_TELEGRAM_CHAT_ID", raising=False)

    _seed_ssm_and_s3(extra_ssm_params={
        "/morning-signal/flow-doctor-telegram-bot-token": "9999:fake-token",
        "/morning-signal/flow-doctor-telegram-chat-id": "8606899594",
    })

    fresh_ge_module._maybe_load_from_ssm()

    assert os.environ["FLOW_DOCTOR_TELEGRAM_BOT_TOKEN"] == "9999:fake-token"
    assert os.environ["FLOW_DOCTOR_TELEGRAM_CHAT_ID"] == "8606899594"


@mock_aws
def test_maybe_load_from_ssm_tolerates_missing_telegram_params(
    fresh_ge_module, aws_env, monkeypatch
):
    """Installs that haven't created the Telegram SSM params yet must
    still boot — the Telegram fetch is optional per-param."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FLOW_DOCTOR_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("FLOW_DOCTOR_TELEGRAM_CHAT_ID", raising=False)

    _seed_ssm_and_s3()  # no telegram params

    # Must NOT raise on the missing Telegram params.
    fresh_ge_module._maybe_load_from_ssm()
    assert "FLOW_DOCTOR_TELEGRAM_BOT_TOKEN" not in os.environ


@mock_aws
def test_maybe_load_from_ssm_respects_local_env_override(
    fresh_ge_module, aws_env, monkeypatch
):
    """If FLOW_DOCTOR_TELEGRAM_BOT_TOKEN is already set in the env (a
    .env file, a one-off shell export), SSM doesn't clobber it. Same
    contract as ANTHROPIC_API_KEY."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.setenv("FLOW_DOCTOR_TELEGRAM_BOT_TOKEN", "local-override-token")
    monkeypatch.delenv("FLOW_DOCTOR_TELEGRAM_CHAT_ID", raising=False)

    _seed_ssm_and_s3(extra_ssm_params={
        "/morning-signal/flow-doctor-telegram-bot-token": "ssm-token",
        "/morning-signal/flow-doctor-telegram-chat-id": "8606899594",
    })

    fresh_ge_module._maybe_load_from_ssm()

    # Local-env override preserved.
    assert os.environ["FLOW_DOCTOR_TELEGRAM_BOT_TOKEN"] == "local-override-token"
    # Chat id (not pre-set locally) was filled from SSM.
    assert os.environ["FLOW_DOCTOR_TELEGRAM_CHAT_ID"] == "8606899594"


@mock_aws
def test_maybe_load_from_ssm_loads_openrouter_key_when_present(
    fresh_ge_module, aws_env, monkeypatch
):
    """The optional OpenRouter key (config#1659 Phase B / shadow-canary use)
    flows into OPENROUTER_API_KEY when the SSM param exists. Not required
    for production generation, which is served by the krepis router (its
    own credentials) once ``llm`` names a router group."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    _seed_ssm_and_s3(extra_ssm_params={
        "/morning-signal/openrouter-api-key": "sk-or-fake-key",
    })

    fresh_ge_module._maybe_load_from_ssm()

    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-fake-key"


@mock_aws
def test_maybe_load_from_ssm_tolerates_missing_openrouter_key(
    fresh_ge_module, aws_env, monkeypatch
):
    """Installs that haven't provisioned the OpenRouter key yet must still
    boot — production generation doesn't need it until Phase B flips."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    _seed_ssm_and_s3()  # no openrouter key seeded

    fresh_ge_module._maybe_load_from_ssm()

    assert "OPENROUTER_API_KEY" not in os.environ


@mock_aws
def test_maybe_load_from_ssm_openrouter_local_override_wins(
    fresh_ge_module, aws_env, monkeypatch
):
    """A pre-set OPENROUTER_API_KEY (local debugging) is not overwritten."""
    monkeypatch.setenv("MORNING_SIGNAL_USE_SSM", "1")
    monkeypatch.setenv("MORNING_SIGNAL_SSM_REGION", REGION_OTHER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "local-override-key")

    _seed_ssm_and_s3(extra_ssm_params={
        "/morning-signal/openrouter-api-key": "ssm-key",
    })

    fresh_ge_module._maybe_load_from_ssm()

    assert os.environ["OPENROUTER_API_KEY"] == "local-override-key"


# ── publish_to_s3 ────────────────────────────────────────────────────────────


def _bootstrap_bucket(name: str = "test-bucket") -> None:
    s3 = boto3.client("s3", region_name=REGION_S3)
    s3.create_bucket(
        Bucket=name, CreateBucketConfiguration={"LocationConstraint": REGION_S3}
    )


@mock_aws
def test_publish_to_s3_uploads_new_mp3s(
    fresh_ge_module, aws_env, sample_config, tmp_episodes_dir, make_episode
):
    _bootstrap_bucket()
    make_episode("2026-05-14", "am")
    # Stage a fake artwork in CWD so the artwork-upload branch fires
    art = Path.cwd() / "artwork.jpg"
    art.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG bytes
    try:
        fresh_ge_module.publish_to_s3(sample_config, fresh_uploads={"2026-05-14-am.mp3"})
    finally:
        art.unlink()

    s3 = boto3.client("s3", region_name=REGION_S3)
    keys = [o["Key"] for o in s3.list_objects_v2(Bucket="test-bucket").get("Contents", [])]
    assert "episodes/2026-05-14-am.mp3" in keys
    assert "feed.xml" in keys


@mock_aws
def test_publish_to_s3_fresh_uploads_always_overwrite(
    fresh_ge_module, aws_env, sample_config, tmp_episodes_dir, make_episode
):
    """F13 fix: a freshly-generated MP3 must overwrite even if S3 already has the key."""
    _bootstrap_bucket()
    make_episode("2026-05-14", "am", audio_bytes=b"NEW" * 500)

    s3 = boto3.client("s3", region_name=REGION_S3)
    # Pre-seed S3 with an old, smaller object at the same key
    s3.put_object(Bucket="test-bucket", Key="episodes/2026-05-14-am.mp3", Body=b"OLD")

    fresh_ge_module.publish_to_s3(sample_config, fresh_uploads={"2026-05-14-am.mp3"})

    obj = s3.get_object(Bucket="test-bucket", Key="episodes/2026-05-14-am.mp3")
    body = obj["Body"].read()
    assert body.startswith(b"NEW"), "fresh upload should overwrite existing S3 object"


@mock_aws
def test_publish_to_s3_skips_existing_back_catalog(
    fresh_ge_module, aws_env, sample_config, tmp_episodes_dir, make_episode
):
    """HEAD-skip applies only to historical MP3s not in fresh_uploads."""
    _bootstrap_bucket()
    make_episode("2026-04-13", "am", audio_bytes=b"BACKCATALOG")

    s3 = boto3.client("s3", region_name=REGION_S3)
    s3.put_object(Bucket="test-bucket", Key="episodes/2026-04-13-am.mp3", Body=b"PRESERVED")

    fresh_ge_module.publish_to_s3(sample_config, fresh_uploads=set())

    obj = s3.get_object(Bucket="test-bucket", Key="episodes/2026-04-13-am.mp3")
    assert obj["Body"].read() == b"PRESERVED"


@mock_aws
def test_publish_to_s3_uploads_missing_back_catalog(
    fresh_ge_module, aws_env, sample_config, tmp_episodes_dir, make_episode
):
    """When S3 doesn't have a back-catalog MP3 yet, it should be uploaded."""
    _bootstrap_bucket()
    make_episode("2026-04-13", "am")
    fresh_ge_module.publish_to_s3(sample_config, fresh_uploads=set())
    s3 = boto3.client("s3", region_name=REGION_S3)
    keys = [o["Key"] for o in s3.list_objects_v2(Bucket="test-bucket").get("Contents", [])]
    assert "episodes/2026-04-13-am.mp3" in keys


# SES-specific notifier tests retired in the 0.5.0rc3 Telegram cutover.
# The notification transport is now flow-doctor's TelegramNotifier; its
# behaviour (POST shape, target-id contract, never-raises semantics,
# preflight) is covered in flow-doctor's own test suite. The morning-
# signal layer's responsibility is the body-construction + main-loop
# wiring, covered by tests/test_notify.py + tests/test_orchestration.py.
