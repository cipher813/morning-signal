"""Tests for the init wizard step functions + scheduler installers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")


# ── step_check_aws ───────────────────────────────────────────────────────────


@mock_aws
def test_step_check_aws_success(aws_env):
    from morning_signal.init.wizard import step_check_aws

    r = step_check_aws()
    assert r.ok
    assert "Account" in r.message or "account" in r.message
    assert "account" in r.detail


def test_step_check_aws_failure_without_creds(monkeypatch):
    from morning_signal.init.wizard import step_check_aws

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent"))

    with patch("boto3.client") as mock_client:
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.side_effect = Exception("NoCredentialsError")
        mock_client.return_value = mock_sts
        r = step_check_aws()
    assert not r.ok
    assert "aws configure" in r.message.lower()


# ── step_check_router ────────────────────────────────────────────────────────


def test_step_check_router_success(monkeypatch):
    from morning_signal.init.wizard import step_check_router

    fake_spec = MagicMock()
    fake_route = {"route": "litellm_proxy"}
    fake_router = MagicMock()
    fake_router.resolve_group_spec.return_value = (fake_spec, fake_route)
    monkeypatch.setitem(sys.modules, "krepis.router", fake_router)

    r = step_check_router()
    assert r.ok
    assert "med" in r.message
    assert r.detail["route"] == fake_route


def test_step_check_router_unreachable(monkeypatch):
    from morning_signal.init.wizard import step_check_router

    fake_router = MagicMock()
    fake_router.resolve_group_spec.side_effect = RuntimeError("no route")
    monkeypatch.setitem(sys.modules, "krepis.router", fake_router)

    r = step_check_router()
    assert not r.ok
    assert "unreachable" in r.message.lower()


# ── step_create_bucket ───────────────────────────────────────────────────────


@mock_aws
def test_step_create_bucket_fresh(aws_env):
    from morning_signal.init.wizard import step_create_bucket

    r = step_create_bucket("test-bucket-fresh", region="us-west-2")
    assert r.ok
    assert r.detail["bucket"] == "test-bucket-fresh"
    assert r.detail["base_url"].startswith("https://test-bucket-fresh.s3.")
    # Confirm the policy + CORS landed
    s3 = boto3.client("s3", region_name="us-west-2")
    policy = json.loads(s3.get_bucket_policy(Bucket="test-bucket-fresh")["Policy"])
    assert policy["Statement"][0]["Action"] == "s3:GetObject"


@mock_aws
def test_step_create_bucket_idempotent(aws_env):
    """Re-running on a bucket we already own should succeed, not fail."""
    from morning_signal.init.wizard import step_create_bucket

    step_create_bucket("idempotent-bucket", region="us-west-2")
    r2 = step_create_bucket("idempotent-bucket", region="us-west-2")
    assert r2.ok


@mock_aws
def test_step_create_bucket_us_east_1_no_constraint(aws_env):
    """us-east-1 rejects LocationConstraint — make sure we omit it."""
    from morning_signal.init.wizard import step_create_bucket

    r = step_create_bucket("east-bucket", region="us-east-1")
    assert r.ok


def test_generate_bucket_suffix_is_lowercase_alnum():
    from morning_signal.init.wizard import generate_bucket_suffix

    suffix = generate_bucket_suffix(length=8)
    assert len(suffix) == 8
    assert suffix.isalnum()
    assert suffix == suffix.lower()


# ── step_write_config ────────────────────────────────────────────────────────


def test_step_write_config_writes_both_files(tmp_path):
    from morning_signal.init.wizard import WizardState, step_write_config

    state = WizardState(
        workdir=tmp_path,
        bucket_name="b",
        bucket_region="us-west-2",
        base_url="https://b.s3.us-west-2.amazonaws.com",
        podcast_title="Test",
        podcast_author="A",
        podcast_email="a@example.com",
        prompt_style="generic-news",
    )
    r = step_write_config(state)
    assert r.ok
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "prompt.md").exists()

    import yaml
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["s3_bucket"] == "b"
    assert cfg["podcast"]["title"] == "Test"
    assert cfg["llm"] == {"provider": "router", "model": "med"}
    assert cfg["fallback_llm"] == {"provider": "router", "model": "low"}


def test_step_write_config_refuses_overwrite_without_force(tmp_path):
    from morning_signal.init.wizard import WizardState, step_write_config

    (tmp_path / "config.yaml").write_text("existing: true")
    state = WizardState(workdir=tmp_path, bucket_name="b", base_url="x")
    r = step_write_config(state, force=False)
    assert not r.ok
    assert "Refusing" in r.message
    assert (tmp_path / "config.yaml").read_text() == "existing: true"  # unchanged


def test_step_write_config_overwrites_with_force(tmp_path):
    from morning_signal.init.wizard import WizardState, step_write_config

    (tmp_path / "config.yaml").write_text("old: data")
    state = WizardState(workdir=tmp_path, bucket_name="b", base_url="x")
    r = step_write_config(state, force=True)
    assert r.ok
    assert "old: data" not in (tmp_path / "config.yaml").read_text()


def test_step_write_config_prompt_style_blank(tmp_path):
    from morning_signal.init.wizard import WizardState, step_write_config

    state = WizardState(workdir=tmp_path, bucket_name="b", base_url="x", prompt_style="blank")
    step_write_config(state)
    # The bundled prompt-blank.md is a stub that explains how to write a prompt.
    assert "Your prompt goes here" in (tmp_path / "prompt.md").read_text()


def test_step_write_config_loads_each_preset(tmp_path):
    """Every PROMPT_PRESETS value should load successfully from morning_signal/data/."""
    from morning_signal.init.wizard import PROMPT_PRESETS, WizardState, step_write_config

    for preset in PROMPT_PRESETS:
        workdir = tmp_path / preset
        workdir.mkdir()
        state = WizardState(workdir=workdir, bucket_name="b", base_url="x", prompt_style=preset)
        r = step_write_config(state)
        assert r.ok, f"preset {preset!r} failed to write: {r.message}"
        content = (workdir / "prompt.md").read_text()
        assert len(content) > 100, f"preset {preset!r} produced suspiciously short prompt"


def test_step_write_config_unknown_preset_falls_back_to_blank(tmp_path):
    from morning_signal.init.wizard import WizardState, step_write_config

    state = WizardState(workdir=tmp_path, bucket_name="b", base_url="x", prompt_style="nonexistent")
    step_write_config(state)
    assert (tmp_path / "prompt.md").exists()


# step_save_anthropic_key retired (alpha-engine-config-I9306): no
# direct-provider credential is collected or persisted any more — see
# step_check_router above.


# ── step_install_scheduler ───────────────────────────────────────────────────


def test_step_install_scheduler_skip(tmp_path):
    from morning_signal.init.wizard import step_install_scheduler

    r = step_install_scheduler("skip", tmp_path)
    assert r.ok
    assert "skipped" in r.message.lower()


def test_step_install_scheduler_invalid_choice(tmp_path):
    from morning_signal.init.wizard import step_install_scheduler

    r = step_install_scheduler("weekly", tmp_path)
    assert not r.ok


def test_step_install_scheduler_launchd_once_daily(tmp_path, monkeypatch):
    """Direct launchd installer test, with launchd_dir redirected to tmp."""
    from morning_signal.init import wizard

    monkeypatch.setattr(wizard, "_launchd_dir", lambda: tmp_path / "LaunchAgents")
    r = wizard.step_install_scheduler("once-daily", tmp_path, kind="launchd")
    assert r.ok
    plist = (tmp_path / "LaunchAgents" / "com.morning-signal.generate.plist").read_text()
    assert "<integer>5</integer>" in plist  # 5 AM
    assert "<integer>17</integer>" not in plist  # not 5 PM
    assert "com.morning-signal.generate" in plist


def test_step_install_scheduler_launchd_twice_daily(tmp_path, monkeypatch):
    from morning_signal.init import wizard

    monkeypatch.setattr(wizard, "_launchd_dir", lambda: tmp_path / "LaunchAgents")
    r = wizard.step_install_scheduler("twice-daily", tmp_path, kind="launchd")
    assert r.ok
    plist = (tmp_path / "LaunchAgents" / "com.morning-signal.generate.plist").read_text()
    assert "<integer>5</integer>" in plist
    assert "<integer>17</integer>" in plist


def test_step_install_scheduler_systemd_twice_daily(tmp_path, monkeypatch):
    from morning_signal.init import wizard

    monkeypatch.setattr(wizard, "_systemd_user_dir", lambda: tmp_path / "systemd-user")
    r = wizard.step_install_scheduler("twice-daily", tmp_path, kind="systemd-user")
    assert r.ok
    service = (tmp_path / "systemd-user" / "morning-signal.service").read_text()
    timer = (tmp_path / "systemd-user" / "morning-signal.timer").read_text()
    assert "ExecStart" in service
    assert "OnCalendar=*-*-* 05:00:00 America/Los_Angeles" in timer
    assert "OnCalendar=*-*-* 17:00:00 America/Los_Angeles" in timer
    assert "Persistent=true" in timer


def test_step_install_scheduler_cron_emits_snippet(tmp_path):
    from morning_signal.init import wizard

    r = wizard.step_install_scheduler("once-daily", tmp_path, kind="cron")
    assert r.ok
    assert "0 5 * * *" in r.detail["snippet"]


def test_step_install_scheduler_none_kind(tmp_path):
    from morning_signal.init import wizard

    r = wizard.step_install_scheduler("once-daily", tmp_path, kind="none")
    assert not r.ok


def test_detect_scheduler_returns_known_value():
    from morning_signal.init.wizard import detect_scheduler

    assert detect_scheduler() in {"launchd", "systemd-user", "cron", "none"}


# ── step_print_subscribe ─────────────────────────────────────────────────────


def test_step_print_subscribe_includes_feed_url():
    from morning_signal.init.wizard import WizardState, step_print_subscribe

    state = WizardState(base_url="https://example.com")
    r = step_print_subscribe(state)
    assert r.ok
    assert "https://example.com/feed.xml" in r.message
    assert "Apple" in r.message


# ── cli init dispatches to wizard.run ────────────────────────────────────────


def test_cli_init_dispatches_to_wizard_run(monkeypatch):
    from typer.testing import CliRunner
    from morning_signal.cli import app

    called = []
    monkeypatch.setattr("morning_signal.init.wizard.run", lambda: called.append(True) or 0)
    result = CliRunner().invoke(app, ["init"])
    assert result.exit_code == 0
    assert called == [True]


def test_cli_init_returns_nonzero_when_wizard_fails(monkeypatch):
    from typer.testing import CliRunner
    from morning_signal.cli import app

    monkeypatch.setattr("morning_signal.init.wizard.run", lambda: 1)
    result = CliRunner().invoke(app, ["init"])
    assert result.exit_code == 1


# ── run() happy-path integration ─────────────────────────────────────────────


@mock_aws
def test_run_happy_path_e2e(tmp_path, aws_env, monkeypatch):
    """Drive run() through a successful flow via monkeypatched prompts.

    Confirms the orchestrator wires the step functions together correctly.
    """
    from morning_signal.init import wizard

    # Mock typer.prompt / typer.confirm by walking a scripted Q&A list.
    prompt_answers = iter(
        [
            "test-bucket-e2e",                                  # bucket name
            "us-west-2",                                        # region
            "Test Podcast",                                     # title
            "Test Author",                                      # author
            "owner@example.com",                                # owner email
            "blank",                                            # prompt style (avoid big string)
            "skip",                                             # schedule choice
        ]
    )
    monkeypatch.setattr(wizard.typer, "prompt", lambda *a, **kw: next(prompt_answers))
    monkeypatch.setattr(wizard.typer, "confirm", lambda *a, **kw: False)  # skip smoke test
    monkeypatch.setattr(wizard.typer, "echo", lambda *a, **kw: None)

    # Mock krepis router to resolve cleanly
    fake_router = MagicMock()
    fake_router.resolve_group_spec.return_value = (MagicMock(), {"route": "litellm_proxy"})
    monkeypatch.setitem(sys.modules, "krepis.router", fake_router)

    code = wizard.run(workdir=tmp_path)
    assert code == 0
    # config.yaml + prompt.md written
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "prompt.md").exists()


@mock_aws
def test_run_aborts_on_aws_failure(tmp_path, monkeypatch):
    """If AWS creds aren't configured, run() returns non-zero early."""
    from morning_signal.init import wizard

    monkeypatch.setattr(wizard, "step_check_aws", lambda: wizard.StepResult(ok=False, message="no creds"))
    monkeypatch.setattr(wizard.typer, "echo", lambda *a, **kw: None)

    assert wizard.run(workdir=tmp_path) == 1


@mock_aws
def test_run_aborts_on_router_failure(tmp_path, aws_env, monkeypatch):
    """If the krepis router is unreachable, run() returns non-zero early."""
    from morning_signal.init import wizard

    monkeypatch.setattr(wizard.typer, "echo", lambda *a, **kw: None)

    fake_router = MagicMock()
    fake_router.resolve_group_spec.side_effect = RuntimeError("unreachable")
    monkeypatch.setitem(sys.modules, "krepis.router", fake_router)

    assert wizard.run(workdir=tmp_path) == 1
