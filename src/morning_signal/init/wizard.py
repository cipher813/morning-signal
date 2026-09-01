"""Interactive `morning-signal init` wizard.

Walks a new user from a fresh clone (or pip install) to a working setup:
AWS creds verified, krepis router connectivity checked, S3 bucket
bootstrapped, config.yaml + prompt.md written, scheduler installed,
smoke-tested. No direct-provider credential is collected or stored
(alpha-engine-config-I9306, Brian's 2026-08-29 ruling): LLM calls route
through the fleet's shared krepis router exclusively.

Architecture: each step is a pure function taking explicit inputs and
returning a result. The `run` orchestrator collects inputs via typer.prompt
and threads results between steps. This keeps the step functions trivial
to unit-test without mocking typer.prompt.
"""

from __future__ import annotations

import json
import platform
import secrets
import shutil
import string
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer
import yaml


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class StepResult:
    ok: bool
    message: str = ""
    detail: Optional[dict] = None


@dataclass
class WizardState:
    """Accumulated state as the wizard walks through steps."""
    workdir: Path = field(default_factory=Path.cwd)
    aws_account_id: str = ""
    aws_user_arn: str = ""
    bucket_name: str = ""
    bucket_region: str = "us-west-2"
    base_url: str = ""
    podcast_title: str = "Morning Signal"
    podcast_author: str = ""
    podcast_email: str = ""
    prompt_style: str = "generic-news"  # generic-news | political-watchlist | blank
    schedule_choice: str = "skip"  # skip | once-daily | twice-daily
    skipped_steps: list = field(default_factory=list)


# ── Step 1: AWS check ────────────────────────────────────────────────────────


def step_check_aws() -> StepResult:
    """Verify AWS creds via sts.get_caller_identity. No mutations."""
    try:
        import boto3
        sts = boto3.client("sts")
        ident = sts.get_caller_identity()
        return StepResult(
            ok=True,
            message=f"AWS creds OK (account {ident['Account']}, identity {ident['Arn']})",
            detail={"account": ident["Account"], "arn": ident["Arn"]},
        )
    except Exception as e:
        return StepResult(
            ok=False,
            message=(
                "AWS credentials not configured. Run `aws configure` (or set "
                "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY env vars) and re-run "
                f"`morning-signal init`.\nError: {e}"
            ),
        )


# ── Step 2: krepis router connectivity ───────────────────────────────────────


def step_check_router() -> StepResult:
    """Verify the krepis router edge can resolve a model group.

    Replaces the pre-migration ``step_check_anthropic`` (validated a
    user-pasted Anthropic API key). Brian's 2026-08-29 ruling
    (alpha-engine-config-I9306/I9263): "we shouldn't be using the anthropic
    api at all" — morning-signal has no direct-provider credential to
    collect any more, so this step confirms the box can reach the
    fleet-shared krepis router instead of prompting for a secret.
    Tests monkeypatch ``krepis.router.resolve_group_spec``.
    """
    try:
        from krepis.router import resolve_group_spec
        _, route = resolve_group_spec("med", max_tokens=64)
    except Exception as e:
        return StepResult(
            ok=False,
            message=(
                "krepis router unreachable — could not resolve model group "
                "'med'. morning-signal has no direct-provider fallback "
                f"(alpha-engine-config-I9306); fix router connectivity "
                f"before continuing.\nError: {e}"
            ),
        )
    resolved_route = route.get("route") if isinstance(route, dict) else route
    return StepResult(
        ok=True,
        message=f"krepis router OK (group 'med' resolved via {resolved_route!r}).",
        detail={"route": route},
    )


# ── Step 3: S3 bucket bootstrap ──────────────────────────────────────────────


def generate_bucket_suffix(length: int = 6) -> str:
    """Random lowercase suffix to avoid global bucket-name collisions."""
    return "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(length))


def step_create_bucket(bucket_name: str, region: str = "us-west-2") -> StepResult:
    """Create the bucket, disable public-access block, set public-read + CORS.

    Idempotent: if the bucket already exists in this account, treat as success.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except Exception as e:
        return StepResult(ok=False, message=f"boto3 import failed: {e}")

    s3 = boto3.client("s3", region_name=region)

    # Create bucket (idempotent).
    try:
        if region == "us-east-1":
            # us-east-1 doesn't accept LocationConstraint and rejects the kwarg.
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("BucketAlreadyOwnedByYou",):
            pass  # idempotent success
        elif code == "BucketAlreadyExists":
            return StepResult(
                ok=False,
                message=(
                    f"Bucket name '{bucket_name}' is taken by someone else. "
                    "S3 bucket names are globally unique — pick another."
                ),
            )
        else:
            return StepResult(ok=False, message=f"Bucket create failed ({code}): {e}")

    # Disable account-level public-access block on the bucket.
    try:
        s3.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": False,
                "IgnorePublicAcls": False,
                "BlockPublicPolicy": False,
                "RestrictPublicBuckets": False,
            },
        )
    except ClientError as e:
        return StepResult(ok=False, message=f"put_public_access_block failed: {e}")

    # Public-read policy on objects (NOT the bucket itself).
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadObjects",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{bucket_name}/*",
            }
        ],
    }
    try:
        s3.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(policy))
    except ClientError as e:
        return StepResult(ok=False, message=f"put_bucket_policy failed: {e}")

    # CORS so podcast apps can fetch.
    cors = {
        "CORSRules": [
            {
                "AllowedOrigins": ["*"],
                "AllowedMethods": ["GET", "HEAD"],
                "AllowedHeaders": ["*"],
                "MaxAgeSeconds": 3600,
            }
        ]
    }
    try:
        s3.put_bucket_cors(Bucket=bucket_name, CORSConfiguration=cors)
    except ClientError as e:
        return StepResult(ok=False, message=f"put_bucket_cors failed: {e}")

    base_url = f"https://{bucket_name}.s3.{region}.amazonaws.com"
    return StepResult(
        ok=True,
        message=f"Bucket s3://{bucket_name} ready in {region}. base_url: {base_url}",
        detail={"bucket": bucket_name, "region": region, "base_url": base_url},
    )


# ── Step 4: Write config + prompt ────────────────────────────────────────────


_PROMPT_PRESET_FILES = {
    "generic-news": "prompt-generic-news.md",
    "tech-only": "prompt-tech-only.md",
    "markets-only": "prompt-markets-only.md",
    "local-news": "prompt-local-news.md",
    "blank": "prompt-blank.md",
}

PROMPT_PRESETS = tuple(_PROMPT_PRESET_FILES.keys())


def _load_preset(style: str) -> str:
    """Read a bundled prompt preset from morning_signal/data/. Falls back to blank."""
    filename = _PROMPT_PRESET_FILES.get(style, _PROMPT_PRESET_FILES["blank"])
    data_dir = Path(__file__).resolve().parent.parent / "data"
    preset_path = data_dir / filename
    if preset_path.exists():
        return preset_path.read_text()
    return f"# {filename} not bundled; please write your prompt here.\n"


def step_write_config(state: WizardState, force: bool = False) -> StepResult:
    """Write config.yaml + prompt.md into the workdir.

    Returns ok=False if either file exists and force is False (caller should
    re-invoke with force=True after confirming).
    """
    config_path = state.workdir / "config.yaml"
    prompt_path = state.workdir / "prompt.md"

    existing = []
    if config_path.exists():
        existing.append("config.yaml")
    if prompt_path.exists():
        existing.append("prompt.md")
    if existing and not force:
        return StepResult(
            ok=False,
            message=f"Refusing to overwrite existing files: {', '.join(existing)}",
        )

    config_data = {
        "s3_bucket": state.bucket_name,
        "s3_region": state.bucket_region,
        "s3_prefix": "",
        "base_url": state.base_url,
        "podcast": {
            "title": state.podcast_title,
            "description": f"{state.podcast_title} — auto-generated daily briefing.",
            "author": state.podcast_author,
            "email": state.podcast_email,
            "language": "en-us",
            "category": "News",
            "subcategory": "Daily News",
            "explicit": False,
            "artwork": "artwork.jpg",
        },
        "tts": {
            "polly_voice": "Ruth",
            "polly_engine": "neural",
            "speed": 1.5,
        },
        # krepis router groups (alpha-engine-config-I9306) — no direct
        # provider credential; the box authenticates to the shared router
        # edge, not this per-instance config. Replaces the pre-migration
        # "claude_model": "claude-sonnet-4-6" default.
        "llm": {"provider": "router", "model": "med"},
        "fallback_llm": {"provider": "router", "model": "low"},
        "max_tokens": 8192,
        "feed_max_episodes": 90,
        "notifications": {
            "enabled": False,
            "sender": "",
            "recipients": [],
            "ses_region": "us-east-1",
        },
    }
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False))
    prompt_path.write_text(_load_preset(state.prompt_style))
    return StepResult(
        ok=True,
        message=f"Wrote {config_path.name} + {prompt_path.name} into {state.workdir}",
        detail={"config_path": str(config_path), "prompt_path": str(prompt_path)},
    )


# ── Step 5: Scheduler installer ───────────────────────────────────────────────


def _systemd_user_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def _launchd_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def detect_scheduler() -> str:
    """Return 'launchd' / 'systemd-user' / 'cron' / 'none' depending on OS support."""
    if platform.system() == "Darwin":
        return "launchd"
    if platform.system() == "Linux":
        if shutil.which("systemctl"):
            return "systemd-user"
        if shutil.which("crontab"):
            return "cron"
    return "none"


def _morning_signal_executable() -> str:
    """Path to the `morning-signal` console script if installed, else python -m."""
    found = shutil.which("morning-signal")
    if found:
        return found
    return f"{sys.executable} -m morning_signal.cli"


def step_install_scheduler(choice: str, workdir: Path, kind: Optional[str] = None) -> StepResult:
    """Install the scheduler.

    choice:
        skip          — no scheduler installed
        once-daily    — fire at 5 AM Pacific
        twice-daily   — fire at 5 AM + 5 PM Pacific
    kind: 'launchd' / 'systemd-user' / 'cron' / 'none' — overrideable for tests.
    """
    if choice == "skip":
        return StepResult(ok=True, message="Scheduler skipped (run manually via `morning-signal generate`).")
    if choice not in ("once-daily", "twice-daily"):
        return StepResult(ok=False, message=f"Unknown schedule choice: {choice!r}")

    kind = kind or detect_scheduler()
    if kind == "none":
        return StepResult(
            ok=False,
            message="No supported scheduler detected (launchd / systemd / cron). Skipping.",
        )

    times_pt = [(5, 0)] if choice == "once-daily" else [(5, 0), (17, 0)]
    exe = _morning_signal_executable()

    if kind == "launchd":
        return _install_launchd(workdir, times_pt, exe)
    if kind == "systemd-user":
        return _install_systemd_user(workdir, times_pt, exe)
    if kind == "cron":
        return _install_cron(workdir, times_pt, exe)
    return StepResult(ok=False, message=f"Unknown scheduler kind: {kind!r}")


def _install_launchd(workdir: Path, times_pt: list, exe: str) -> StepResult:
    plist_dir = _launchd_dir()
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / "com.morning-signal.generate.plist"
    cal_entries = "\n".join(
        f"            <dict><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>{m}</integer></dict>"
        for h, m in times_pt
    )
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.morning-signal.generate</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>cd {workdir} && {exe} generate</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>
{cal_entries}
    </array>
    <key>StandardOutPath</key><string>{workdir}/cron.log</string>
    <key>StandardErrorPath</key><string>{workdir}/cron.log</string>
</dict>
</plist>
"""
    plist_path.write_text(plist)
    return StepResult(
        ok=True,
        message=(
            f"launchd plist written to {plist_path}.\n"
            "Activate: launchctl load ~/Library/LaunchAgents/com.morning-signal.generate.plist\n"
            "NOTE: launchd only fires when the Mac is awake — for reliable daily delivery, "
            "deploy on a long-lived host (see README 'Cloud deploy')."
        ),
        detail={"plist_path": str(plist_path)},
    )


def _install_systemd_user(workdir: Path, times_pt: list, exe: str) -> StepResult:
    unit_dir = _systemd_user_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)
    service = f"""[Unit]
Description=Morning Signal podcast generator

[Service]
Type=oneshot
WorkingDirectory={workdir}
ExecStart=/bin/sh -c "{exe} generate"
TimeoutStartSec=600
"""
    cal_lines = "\n".join(
        f"OnCalendar=*-*-* {h:02d}:{m:02d}:00 America/Los_Angeles" for h, m in times_pt
    )
    timer = f"""[Unit]
Description=Morning Signal — DST-aware Pacific schedule

[Timer]
Unit=morning-signal.service
{cal_lines}
Persistent=true

[Install]
WantedBy=timers.target
"""
    (unit_dir / "morning-signal.service").write_text(service)
    (unit_dir / "morning-signal.timer").write_text(timer)
    return StepResult(
        ok=True,
        message=(
            f"systemd user units written to {unit_dir}.\n"
            "Activate: systemctl --user daemon-reload && systemctl --user enable --now morning-signal.timer"
        ),
        detail={"unit_dir": str(unit_dir)},
    )


def _install_cron(workdir: Path, times_pt: list, exe: str) -> StepResult:
    lines = [
        f"{m} {h} * * * cd {workdir} && {exe} generate >> {workdir}/cron.log 2>&1"
        for h, m in times_pt
    ]
    snippet = "\n".join(lines) + "\n"
    return StepResult(
        ok=True,
        message=(
            "Add the following to your crontab (`crontab -e`):\n\n"
            + snippet
            + "\nNote: cron uses the system local time, not Pacific — adjust hours "
            "if your host isn't in PT."
        ),
        detail={"snippet": snippet},
    )


# ── Step 7: Subscribe instructions ───────────────────────────────────────────


def step_print_subscribe(state: WizardState) -> StepResult:
    feed_url = f"{state.base_url.rstrip('/')}/feed.xml"
    return StepResult(
        ok=True,
        message=(
            f"Feed URL: {feed_url}\n\n"
            "Subscribe in Apple Podcasts (iOS):\n"
            "  Library → ··· top-right → 'Follow a Show by URL…' → paste the feed URL.\n\n"
            "Overcast / Pocket Casts: 'Add URL' with the same feed URL.\n\n"
            "First episode: `morning-signal generate`"
        ),
        detail={"feed_url": feed_url},
    )


# ── Wizard orchestrator ──────────────────────────────────────────────────────


def run(workdir: Optional[Path] = None) -> int:
    """Run the interactive wizard. Returns 0 on success, non-zero on aborted setup."""
    state = WizardState(workdir=workdir or Path.cwd())

    typer.echo("morning-signal init — interactive setup wizard")
    typer.echo("=" * 50)
    typer.echo("")

    # 1. AWS check
    typer.echo("→ Step 1/7: Checking AWS credentials…")
    r = step_check_aws()
    typer.echo(f"  {r.message}")
    if not r.ok:
        return 1
    state.aws_account_id = r.detail["account"]
    state.aws_user_arn = r.detail["arn"]

    # 2. krepis router connectivity
    typer.echo("")
    typer.echo("→ Step 2/7: krepis router connectivity")
    r = step_check_router()
    typer.echo(f"  {r.message}")
    if not r.ok:
        return 1

    # 3. S3 bucket
    typer.echo("")
    typer.echo("→ Step 3/7: S3 bucket")
    default_bucket = f"morning-signal-{generate_bucket_suffix()}"
    bucket_name = typer.prompt(f"  Bucket name", default=default_bucket)
    region = typer.prompt("  Region", default="us-west-2")
    r = step_create_bucket(bucket_name, region)
    typer.echo(f"  {r.message}")
    if not r.ok:
        return 1
    state.bucket_name = bucket_name
    state.bucket_region = region
    state.base_url = r.detail["base_url"]

    # 4. Podcast metadata
    typer.echo("")
    typer.echo("→ Step 4/7: Podcast metadata")
    state.podcast_title = typer.prompt("  Podcast title", default="Morning Signal")
    state.podcast_author = typer.prompt("  Author name", default="")
    state.podcast_email = typer.prompt("  Owner email (for iTunes; optional)", default="")
    style_help = " | ".join(PROMPT_PRESETS)
    state.prompt_style = typer.prompt(
        f"  Prompt style [{style_help}]", default="generic-news"
    )
    if state.prompt_style not in PROMPT_PRESETS:
        typer.echo(f"  Unknown preset {state.prompt_style!r}; falling back to 'blank'.")
        state.prompt_style = "blank"
    r = step_write_config(state, force=False)
    if not r.ok and "Refusing" in r.message:
        if typer.confirm(f"  {r.message}\n  Overwrite?", default=False):
            r = step_write_config(state, force=True)
        else:
            typer.echo("  Skipping config write.")
            r = StepResult(ok=True, message="Existing files kept.")
    typer.echo(f"  {r.message}")
    if not r.ok:
        return 1

    # 5. Scheduler
    typer.echo("")
    typer.echo("→ Step 5/7: Scheduler")
    kind = detect_scheduler()
    typer.echo(f"  Detected scheduler: {kind}")
    if kind == "none":
        typer.echo("  No supported scheduler available; skipping.")
        state.skipped_steps.append("scheduler")
    else:
        choice = typer.prompt(
            "  Install schedule? [skip | once-daily | twice-daily]", default="skip"
        )
        r = step_install_scheduler(choice, state.workdir, kind)
        typer.echo(f"  {r.message}")
        state.schedule_choice = choice

    # 6. Smoke test offer
    typer.echo("")
    typer.echo("→ Step 6/7: Smoke test")
    if typer.confirm("  Run a smoke test now (`morning-signal generate --script-only`)?", default=True):
        try:
            subprocess.run(
                [sys.executable, "-m", "morning_signal.cli", "generate", "--script-only"],
                cwd=state.workdir,
                check=False,
            )
        except Exception as e:
            typer.echo(f"  Smoke test failed: {e}")
    else:
        state.skipped_steps.append("smoke-test")

    # 7. Subscribe
    typer.echo("")
    typer.echo("→ Step 7/7: Subscribe")
    r = step_print_subscribe(state)
    typer.echo(r.message)

    typer.echo("")
    typer.echo("✓ Setup complete.")
    return 0
