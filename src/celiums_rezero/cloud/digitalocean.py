"""Fail-safe single-Droplet DigitalOcean campaign executor."""

from __future__ import annotations

import base64
import json
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Protocol

CLEANUP_RESERVE_SECONDS = 300
ROCM_PYTORCH_IMAGE = "rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1"
BUNDLE_SHA256 = "93db742ead71c12fa46c62661b12108fdb0a815d3b5fcf180821538dcfc8b9be"
REZERO_BUNDLE_SHA256 = "10da3a479058cc94967f510fcbf979af759cf9ca11e18bec1209312606dfe670"
REZERO_V2_BUNDLE_SHA256 = "5ea905ede3ec42bb756714885aca42567655dbb4c4ebdd38029d11345872ca3c"
REZERO_V4_BUNDLE_SHA256 = "5697dda245fe93c19e36a7741c7e6e484b770a426be4303843127bc1444cd121"
NAVIGATION_BUNDLE_SHA256 = "5cb0381c03f944706819e6c5ce2d9dc71be63c27b88292cfd28fd2b489d7b7c8"
NAVIGATION_V2_BUNDLE_SHA256 = "d14963e7835a81ee4ca32274d34ba5ed098270a626ba34690fb706f3465ab7ac"
NAVIGATION_V2_CHECKPOINT_SHA256 = "0f1f140d683df581020a39b221802f20a14ced6d4316748f70aab36ced686844"
NAVIGATION_V3_BUNDLE_SHA256 = "NAV3_BUNDLE_SHA256"
NAVIGATION_V3_CHECKPOINT_SHA256 = "NAV3_CHECKPOINT_SHA256"
HYPHAE_WHEEL_SHA256 = "fd6503abbcac18db9a6705682b80a83904389f146e6dd0c4d17fdef49535a5fb"
HYPHAE_WHEEL_BYTES = 87_754
UNIFIED_CAMPAIGN = "canary-hyphae-minilm-gemma-v1"
REZERO_UNIFIED_CAMPAIGN = "canary-hyphae-minilm-gemma-rezero-v1"
NAVIGATION_UNIFIED_CAMPAIGN = "canary-hyphae-minilm-gemma-navigation-v1"
NAVIGATION_V2_CAMPAIGN = "canary-hyphae-minilm-gemma-navigation-v2"
NAVIGATION_V3_CAMPAIGN = "canary-hyphae-minilm-gemma-navigation-v3"
UNIFIED_EVIDENCE = (
    "unified-campaign-report.json",
    "minilm-preflight.json",
    "gemma4-e4b-preflight.json",
    "protocol-request.json",
    "protocol-response.json",
    "hyphae-daemon.stdout.log",
    "hyphae-daemon.stderr.log",
    "source-revision.txt",
    "python-freeze.txt",
)
NAVIGATION_EVIDENCE = (
    "navigation-campaign-report.json",
    "minilm-preflight.json",
    "gemma4-e4b-preflight.json",
    "hyphae-daemon.stdout.log",
    "hyphae-daemon.stderr.log",
    "source-revision.txt",
    "python-freeze.txt",
)
UNIFIED_CAMPAIGNS = (
    UNIFIED_CAMPAIGN,
    REZERO_UNIFIED_CAMPAIGN,
    NAVIGATION_UNIFIED_CAMPAIGN,
    NAVIGATION_V2_CAMPAIGN,
    NAVIGATION_V3_CAMPAIGN,
)


class CommandRunner(Protocol):
    def run(
        self,
        command: list[str],
        *,
        capture_output: bool = True,
        check: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


class SubprocessRunner:
    def run(
        self,
        command: list[str],
        *,
        capture_output: bool = True,
        check: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            capture_output=capture_output,
            check=check,
            text=True,
            timeout=timeout,
        )


@dataclass(frozen=True, slots=True)
class CloudCampaignPlan:
    name: str
    region: str
    size: str
    image: str
    ssh_key_id: str
    ssh_private_key: Path
    repository_url: str
    revision: str
    data_command: tuple[str, ...]
    campaign_command: tuple[str, ...]
    artifact_directory: Path
    hourly_rate_usd: float
    max_lifetime_seconds: int
    max_cost_usd: float
    accelerator: str = "nvidia"
    remote_root: str = "/opt/hyphae-transformer"
    remote_data_root: str = "/opt/celiums-data"
    remote_run_root: str = "/opt/celiums-runs/campaign"
    hyphae_sdk_wheel: Path | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.region or not self.size or not self.image:
            raise ValueError("cloud resource identifiers are required")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("an immutable source revision is required")
        if self.max_lifetime_seconds < 60:
            raise ValueError("cloud lifetime must be at least 60 seconds")
        if self.max_lifetime_seconds <= CLEANUP_RESERVE_SECONDS:
            raise ValueError("cloud lifetime must exceed the cleanup reserve")
        if min(self.hourly_rate_usd, self.max_cost_usd) <= 0:
            raise ValueError("cloud prices and cost budget must be positive")
        projected_cost = self.hourly_rate_usd * self.max_lifetime_seconds / 3600
        if projected_cost > self.max_cost_usd + 1e-12:
            raise ValueError("maximum lifetime exceeds the cloud cost budget")
        if not self.data_command or self.data_command[0] not in {
            "prepare-data",
            "prepare-gemma4-e4b",
        }:
            raise ValueError("data command must use the prepare-data allowlist")
        if not self.campaign_command or self.campaign_command[0] not in {
            "pilot-wikitext2",
            "pilot-enwiki8",
            "smoke-gemma4-e4b",
            "smoke-gemma4-e4b-rezero-v1",
            "train-gemma4-e4b",
            "train-gemma4-e4b-v2",
            "train-gemma4-e4b-v3",
            "train-gemma4-e4b-rezero-v1",
            "train-gemma4-e4b-rezero-v2",
            "train-gemma4-e4b-rezero-v3",
            "train-gemma4-e4b-rezero-v4",
            "train-gemma4-e4b-rezero-navigation-v1",
            "train-gemma4-e4b-rezero-navigation-v2",
            "train-gemma4-e4b-rezero-navigation-v3",
            "shadow-gemma4-e4b-v1",
            "shadow-gemma4-e4b-v2",
            "shadow-gemma4-e4b-rezero-v1",
            "shadow-gemma4-e4b-rezero-v2",
            "shadow-gemma4-e4b-rezero-v4",
            "canary-gemma4-e4b-quoted-runtime-v1",
            UNIFIED_CAMPAIGN,
            REZERO_UNIFIED_CAMPAIGN,
            NAVIGATION_UNIFIED_CAMPAIGN,
            NAVIGATION_V2_CAMPAIGN,
            NAVIGATION_V3_CAMPAIGN,
        }:
            raise ValueError("campaign command is not allowlisted")
        if self.accelerator not in {"nvidia", "amd-rocm"}:
            raise ValueError("cloud accelerator is not allowlisted")
        gemma_workload = self.data_command[0] == "prepare-gemma4-e4b" or (
            self.campaign_command[0]
            in {
                "smoke-gemma4-e4b",
                "smoke-gemma4-e4b-rezero-v1",
                "train-gemma4-e4b",
                "train-gemma4-e4b-v2",
                "train-gemma4-e4b-v3",
                "train-gemma4-e4b-rezero-v1",
                "train-gemma4-e4b-rezero-v2",
                "train-gemma4-e4b-rezero-v3",
                "train-gemma4-e4b-rezero-v4",
                "train-gemma4-e4b-rezero-navigation-v1",
                "train-gemma4-e4b-rezero-navigation-v2",
                "train-gemma4-e4b-rezero-navigation-v3",
                "shadow-gemma4-e4b-v1",
                "shadow-gemma4-e4b-v2",
                "shadow-gemma4-e4b-rezero-v1",
                "shadow-gemma4-e4b-rezero-v2",
                "shadow-gemma4-e4b-rezero-v4",
                "canary-gemma4-e4b-quoted-runtime-v1",
                UNIFIED_CAMPAIGN,
                REZERO_UNIFIED_CAMPAIGN,
                NAVIGATION_UNIFIED_CAMPAIGN,
                NAVIGATION_V2_CAMPAIGN,
                NAVIGATION_V3_CAMPAIGN,
            }
        )
        if gemma_workload != (self.accelerator == "amd-rocm"):
            raise ValueError("Gemma E4B workload requires the AMD ROCm executor")
        if self.accelerator == "amd-rocm" and (
            self.region != "mem1"
            or self.size != "gpu-mi355x1-288gb-spot"
            or self.image != "amddevelopercloud-pytorch2100rocm724"
        ):
            raise ValueError("Gemma E4B smoke requires the preregistered MI355X x1 shape")
        if self.data_command[0] == "prepare-gemma4-e4b" and self.data_command != (
            "prepare-gemma4-e4b",
        ):
            raise ValueError("Gemma E4B preparation does not accept plan arguments")
        if self.campaign_command[0] == "smoke-gemma4-e4b":
            _validate_gemma_smoke_command(self.campaign_command)
        if self.campaign_command[0] == "smoke-gemma4-e4b-rezero-v1":
            _validate_gemma_rezero_smoke_command(self.campaign_command)
        if self.campaign_command[0] == "train-gemma4-e4b":
            _validate_gemma_training_command(self.campaign_command)
        if self.campaign_command[0] == "train-gemma4-e4b-v2":
            _validate_gemma_v2_training_command(self.campaign_command)
        if self.campaign_command[0] == "train-gemma4-e4b-v3":
            _validate_gemma_v3_training_command(self.campaign_command)
        if self.campaign_command[0] == "train-gemma4-e4b-rezero-v1":
            _validate_gemma_rezero_training_command(self.campaign_command)
        if self.campaign_command[0] == "train-gemma4-e4b-rezero-v2":
            _validate_gemma_rezero_v2_training_command(self.campaign_command)
        if self.campaign_command[0] == "train-gemma4-e4b-rezero-v3":
            _validate_gemma_rezero_v3_training_command(self.campaign_command)
        if self.campaign_command[0] == "train-gemma4-e4b-rezero-v4":
            _validate_gemma_rezero_v4_training_command(self.campaign_command)
        if self.campaign_command[
            0
        ] == "train-gemma4-e4b-rezero-navigation-v1" and self.campaign_command != (
            "train-gemma4-e4b-rezero-navigation-v1",
        ):
            raise ValueError("ReZero navigation command differs from preregistration")
        if self.campaign_command[
            0
        ] == "train-gemma4-e4b-rezero-navigation-v2" and self.campaign_command != (
            "train-gemma4-e4b-rezero-navigation-v2",
            "--calibration",
            "0.03278688524590164",
        ):
            raise ValueError("ReZero navigation v2 command differs from preregistration")
        if self.campaign_command[0] == "shadow-gemma4-e4b-v1":
            _validate_gemma_shadow_command(self.campaign_command)
        if self.campaign_command[0] == "shadow-gemma4-e4b-v2":
            _validate_gemma_shadow_v2_command(self.campaign_command)
        if self.campaign_command[0] == "shadow-gemma4-e4b-rezero-v1":
            _validate_gemma_rezero_shadow_command(self.campaign_command)
        if self.campaign_command[0] == "shadow-gemma4-e4b-rezero-v2":
            _validate_gemma_rezero_v2_shadow_command(self.campaign_command)
        if self.campaign_command[0] == "shadow-gemma4-e4b-rezero-v4":
            _validate_gemma_rezero_v4_shadow_command(self.campaign_command)
        if self.campaign_command[0] == "canary-gemma4-e4b-quoted-runtime-v1":
            _validate_gemma_quoted_runtime_command(self.campaign_command)
        if self.campaign_command[0] in UNIFIED_CAMPAIGNS and (
            self.campaign_command != (self.campaign_command[0],) or self.hyphae_sdk_wheel is None
        ):
            raise ValueError("unified campaign requires one exact Hyphae SDK wheel")
        for value in (
            self.remote_root,
            self.remote_data_root,
            self.remote_run_root,
        ):
            if not value.startswith("/opt/") or any(character.isspace() for character in value):
                raise ValueError("remote paths must be whitespace-free paths under /opt")


@dataclass(frozen=True, slots=True)
class CloudExecutionSummary:
    status: str
    droplet_id: int | None
    droplet_name: str
    public_ip: str | None
    created_at: str | None
    deleted_at: str | None
    lifetime_seconds: float
    estimated_cost_usd: float
    artifact_directory: str
    failure: str | None = None
    dry_run_commands: tuple[tuple[str, ...], ...] = ()


def execute_digitalocean_campaign(
    plan: CloudCampaignPlan,
    *,
    runner: CommandRunner | None = None,
    dry_run: bool = False,
    sleep: object = time.sleep,
) -> CloudExecutionSummary:
    command_runner = SubprocessRunner() if runner is None else runner
    commands = planned_commands(plan)
    if dry_run:
        return CloudExecutionSummary(
            status="dry_run",
            droplet_id=None,
            droplet_name=plan.name,
            public_ip=None,
            created_at=None,
            deleted_at=None,
            lifetime_seconds=0.0,
            estimated_cost_usd=0.0,
            artifact_directory=str(plan.artifact_directory),
            dry_run_commands=tuple(tuple(command) for command in commands),
        )

    if plan.artifact_directory.exists() and any(plan.artifact_directory.iterdir()):
        raise FileExistsError("cloud artifact directory is not empty")
    if runner is None:
        _verify_remote_revision(plan)
    if plan.campaign_command[0] in UNIFIED_CAMPAIGNS:
        _validate_hyphae_wheel(plan)
    droplet_id: int | None = None
    public_ip: str | None = None
    created_at: datetime | None = None
    created_clock: float | None = None
    failure: str | None = None
    status = "failed"
    try:
        create_started_at = datetime.now(UTC)
        create_started_clock = time.monotonic()
        created = command_runner.run(commands[0], timeout=min(600, plan.max_lifetime_seconds))
        values = json.loads(created.stdout)
        if not isinstance(values, list) or len(values) != 1:
            raise RuntimeError("DigitalOcean create did not return exactly one Droplet")
        droplet = values[0]
        droplet_id = int(droplet["id"])
        public_ip = _public_ipv4(droplet)
        created_at = create_started_at
        created_clock = create_started_clock
        _wait_for_ssh(
            plan,
            public_ip,
            command_runner,
            timeout_seconds=min(
                300,
                _remaining_lifetime(plan, created_clock, reserve_seconds=CLEANUP_RESERVE_SECONDS),
            ),
            sleep=sleep,
        )
        if plan.campaign_command[0] in UNIFIED_CAMPAIGNS:
            wheel_upload = command_runner.run(
                _wheel_upload_command(plan, public_ip),
                timeout=min(
                    120,
                    _remaining_lifetime(
                        plan, created_clock, reserve_seconds=CLEANUP_RESERVE_SECONDS
                    ),
                ),
            )
            _write_process_evidence(plan, "wheel-upload", wheel_upload)
        try:
            bootstrap = command_runner.run(
                _ssh_command(plan, public_ip, _bootstrap_script(plan)),
                timeout=_remaining_lifetime(
                    plan, created_clock, reserve_seconds=CLEANUP_RESERVE_SECONDS
                ),
            )
        except subprocess.CalledProcessError as error:
            _write_failed_process_evidence(plan, "bootstrap", error)
            raise
        _write_process_evidence(plan, "bootstrap", bootstrap)
        try:
            campaign = command_runner.run(
                _ssh_command(plan, public_ip, _campaign_script(plan)),
                timeout=_remaining_lifetime(
                    plan, created_clock, reserve_seconds=CLEANUP_RESERVE_SECONDS
                ),
            )
        except subprocess.CalledProcessError as error:
            _write_failed_process_evidence(plan, "campaign", error)
            raise
        _write_process_evidence(plan, "campaign", campaign)
        _validate_remote_source_patch(plan, public_ip, command_runner)
        plan.artifact_directory.mkdir(parents=True, exist_ok=True)
        if plan.campaign_command[0] in {NAVIGATION_V2_CAMPAIGN, NAVIGATION_V3_CAMPAIGN}:
            _write_retrieved_evidence(plan, campaign)
        retrieval = command_runner.run(
            _artifact_command(plan, public_ip),
            timeout=min(
                900,
                _remaining_lifetime(plan, created_clock, reserve_seconds=CLEANUP_RESERVE_SECONDS),
            ),
        )
        if plan.accelerator == "amd-rocm" and plan.campaign_command[0] not in {
            NAVIGATION_V2_CAMPAIGN,
            NAVIGATION_V3_CAMPAIGN,
        }:
            _write_retrieved_evidence(plan, retrieval)
        status = "completed"
    except Exception as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = f": {error.stderr.strip()}"
        failure = f"{type(error).__name__}: {error}{detail}"
        if public_ip is not None:
            try:
                if created_clock is None:
                    raise RuntimeError("cloud creation clock is absent")
                plan.artifact_directory.mkdir(parents=True, exist_ok=True)
                retrieval = command_runner.run(
                    _artifact_command(plan, public_ip),
                    check=False,
                    timeout=min(
                        300,
                        _remaining_lifetime(
                            plan, created_clock, reserve_seconds=CLEANUP_RESERVE_SECONDS
                        ),
                    ),
                )
                if plan.accelerator == "amd-rocm" and retrieval.returncode == 0:
                    _write_retrieved_evidence(plan, retrieval)
            except Exception:
                pass
    finally:
        if droplet_id is not None:
            try:
                _delete_and_verify_droplet(command_runner, droplet_id, sleep=sleep)
            except Exception as error:
                status = "failed"
                deletion_failure = f"DropletDeletionError: {error}"
                failure = deletion_failure if failure is None else f"{failure}; {deletion_failure}"
        deleted_at = datetime.now(UTC)

    lifetime = 0.0 if created_at is None else (deleted_at - created_at).total_seconds()
    estimated_cost = lifetime * plan.hourly_rate_usd / 3600
    if estimated_cost > plan.max_cost_usd and failure is None:
        status = "failed"
        failure = (
            f"BudgetExceeded: estimated cloud cost {estimated_cost:.6f} "
            f"exceeds {plan.max_cost_usd:.6f}"
        )
    summary = CloudExecutionSummary(
        status=status,
        droplet_id=droplet_id,
        droplet_name=plan.name,
        public_ip=public_ip,
        created_at=None if created_at is None else created_at.isoformat(),
        deleted_at=deleted_at.isoformat(),
        lifetime_seconds=lifetime,
        estimated_cost_usd=estimated_cost,
        artifact_directory=str(plan.artifact_directory),
        failure=failure,
    )
    plan.artifact_directory.mkdir(parents=True, exist_ok=True)
    (plan.artifact_directory / "cloud-execution.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary


def planned_commands(plan: CloudCampaignPlan) -> list[list[str]]:
    placeholder = "<PUBLIC_IP>"
    commands = [
        [
            "doctl",
            "compute",
            "droplet",
            "create",
            plan.name,
            "--size",
            plan.size,
            "--image",
            plan.image,
            "--region",
            plan.region,
            "--ssh-keys",
            plan.ssh_key_id,
            "--tag-names",
            "hyphae-transformer,training,gpu,ephemeral",
            "--enable-monitoring",
            "--wait",
            "--output",
            "json",
        ],
    ]
    if plan.campaign_command[0] in UNIFIED_CAMPAIGNS:
        commands.append(_wheel_upload_command(plan, placeholder))
    commands.extend(
        [
            _ssh_command(plan, placeholder, _bootstrap_script(plan)),
            _ssh_command(plan, placeholder, _campaign_script(plan)),
            _artifact_command(plan, placeholder),
            ["doctl", "compute", "droplet", "delete", "<DROPLET_ID>", "--force"],
        ]
    )
    return commands


def _wheel_upload_command(plan: CloudCampaignPlan, public_ip: str) -> list[str]:
    if plan.hyphae_sdk_wheel is None:
        raise ValueError("unified campaign has no Hyphae SDK wheel")
    return [
        "scp",
        "-i",
        str(plan.ssh_private_key),
        "-o",
        "StrictHostKeyChecking=accept-new",
        str(plan.hyphae_sdk_wheel),
        f"root@{public_ip}:/tmp/hyphae_sdk-2.1.0-py3-none-any.whl",
    ]


def _wait_for_ssh(
    plan: CloudCampaignPlan,
    public_ip: str,
    runner: CommandRunner,
    *,
    timeout_seconds: float = 300,
    sleep: object,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = runner.run(
            _ssh_command(plan, public_ip, "printf __HYPHAE_READY__"),
            check=False,
            timeout=20,
        )
        if result.returncode == 0 and result.stdout.strip() == "__HYPHAE_READY__":
            return
        if callable(sleep):
            sleep(min(10, max(0, deadline - time.monotonic())))
    raise TimeoutError("Droplet SSH did not become ready")


def _ssh_command(plan: CloudCampaignPlan, public_ip: str, script: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(plan.ssh_private_key),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=20",
        f"root@{public_ip}",
        script,
    ]


def _bootstrap_script(plan: CloudCampaignPlan) -> str:
    checkout = shlex.join(
        [
            "git",
            "clone",
            "--no-checkout",
            plan.repository_url,
            plan.remote_root,
        ]
    )
    common = (
        f"rm -rf {shlex.quote(plan.remote_root)}",
        checkout,
        (
            f"git -C {shlex.quote(plan.remote_root)} fetch --depth 1 origin "
            f"{shlex.quote(plan.revision)}"
        ),
        f"git -C {shlex.quote(plan.remote_root)} checkout {shlex.quote(plan.revision)}",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"mkdir -p {shlex.quote(plan.remote_data_root)} {shlex.quote(plan.remote_run_root)}",
    )
    if plan.accelerator == "amd-rocm":
        unified: tuple[str, ...] = ()
        if plan.campaign_command[0] in UNIFIED_CAMPAIGNS:
            unified = (
                (
                    "mv /tmp/hyphae_sdk-2.1.0-py3-none-any.whl "
                    f"{shlex.quote(plan.remote_data_root)}/hyphae_sdk-2.1.0-py3-none-any.whl"
                ),
            )
        container = _rocm_container_command(plan)
        return " && ".join(
            (
                *common,
                *unified,
                "rocm-smi --showproductname --showmeminfo vram --showdriverversion",
                f"docker pull {shlex.quote(ROCM_PYTORCH_IMAGE)}",
                f"{container} /bin/bash -lc {shlex.quote(_rocm_bootstrap_inner(plan))}",
            )
        )
    return " && ".join(
        (
            *common,
            f"cd {shlex.quote(plan.remote_root)}",
            "/root/.local/bin/uv sync --frozen",
            shlex.join(
                [
                    "/root/.local/bin/uv",
                    "run",
                    "hyphae-transformer",
                    *plan.data_command,
                    "--root",
                    plan.remote_data_root,
                ]
            ),
            "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
        )
    )


def _campaign_script(plan: CloudCampaignPlan) -> str:
    if plan.campaign_command[0] in {
        "smoke-gemma4-e4b",
        "smoke-gemma4-e4b-rezero-v1",
        "train-gemma4-e4b",
        "train-gemma4-e4b-v2",
        "train-gemma4-e4b-v3",
        "train-gemma4-e4b-rezero-v1",
        "train-gemma4-e4b-rezero-v2",
        "train-gemma4-e4b-rezero-v3",
        "train-gemma4-e4b-rezero-v4",
        "train-gemma4-e4b-rezero-navigation-v1",
        "train-gemma4-e4b-rezero-navigation-v2",
        "train-gemma4-e4b-rezero-navigation-v3",
        "shadow-gemma4-e4b-v1",
        "shadow-gemma4-e4b-v2",
        "shadow-gemma4-e4b-rezero-v1",
        "shadow-gemma4-e4b-rezero-v2",
        "shadow-gemma4-e4b-rezero-v4",
        "canary-gemma4-e4b-quoted-runtime-v1",
        UNIFIED_CAMPAIGN,
        REZERO_UNIFIED_CAMPAIGN,
        NAVIGATION_UNIFIED_CAMPAIGN,
        NAVIGATION_V2_CAMPAIGN,
        NAVIGATION_V3_CAMPAIGN,
    }:
        campaign_seconds = plan.max_lifetime_seconds - CLEANUP_RESERVE_SECONDS
        if plan.campaign_command[0] == "smoke-gemma4-e4b":
            command = [
                "python",
                "/workspace/scripts/smoke_gemma4_e4b.py",
                "--model",
                "/data/gemma4-e4b",
                "--dataset",
                "/workspace/experiments/governed/mars-v2-e4b-v1",
                "--out",
                "/runs/gemma4-e4b-smoke.json",
                *plan.campaign_command[1:],
            ]
        elif plan.campaign_command[0] == "smoke-gemma4-e4b-rezero-v1":
            command = [
                "python",
                "/workspace/scripts/smoke_gemma4_e4b_rezero_control.py",
                "--model",
                "/data/gemma4-e4b",
                "--dataset",
                "/workspace/experiments/governed/mars-v2-e4b-v1",
                "--preregistration",
                ("/workspace/experiments/canonical/gemma4_e4b_rezero_sequence_control_v1.json"),
                "--out",
                "/runs/gemma4-e4b-rezero-smoke.json",
                *plan.campaign_command[1:],
            ]
        elif plan.campaign_command[0] in {
            "train-gemma4-e4b-rezero-v1",
            "train-gemma4-e4b-rezero-v2",
            "train-gemma4-e4b-rezero-v3",
            "train-gemma4-e4b-rezero-v4",
        }:
            rezero_version = plan.campaign_command[0].rsplit("-", 1)[-1]
            command = [
                "python",
                "/workspace/scripts/train_gemma4_e4b_rezero_control.py",
                "--model",
                "/data/gemma4-e4b",
                "--dataset",
                "/workspace/experiments/governed/mars-v2-e4b-v1",
                "--preregistration",
                (
                    "/workspace/experiments/canonical/"
                    f"gemma4_e4b_rezero_sequence_control_{rezero_version}.json"
                ),
                "--out",
                "/runs",
                *plan.campaign_command[1:],
            ]
        elif plan.campaign_command[0] == "train-gemma4-e4b-rezero-navigation-v1":
            command = [
                "python",
                "/workspace/scripts/train_gemma4_e4b_navigation.py",
                "--model",
                "/data/gemma4-e4b",
                "--dataset",
                "/workspace/experiments/governed/mars-v2-e4b-v1",
                "--preregistration",
                ("/workspace/experiments/canonical/gemma4_e4b_rezero_navigation_v1.json"),
                "--out",
                "/runs",
                *plan.campaign_command[1:],
            ]
        elif plan.campaign_command[0] == "train-gemma4-e4b-rezero-navigation-v3":
            command = [
                "python",
                "/workspace/scripts/train_gemma4_e4b_navigation_v3.py",
                "--model",
                "/data/gemma4-e4b",
                "--dataset",
                "/workspace/experiments/governed/mars-v2-e4b-v1",
                "--preregistration",
                ("/workspace/experiments/canonical/gemma4_e4b_rezero_navigation_v3.json"),
                "--out",
                "/runs",
                *plan.campaign_command[1:],
            ]
        elif plan.campaign_command[0] == "train-gemma4-e4b-rezero-navigation-v2":
            command = [
                "python",
                "/workspace/scripts/train_gemma4_e4b_navigation_v2.py",
                "--model",
                "/data/gemma4-e4b",
                "--dataset",
                "/workspace/experiments/governed/mars-v2-e4b-v1",
                "--preregistration",
                ("/workspace/experiments/canonical/gemma4_e4b_rezero_navigation_v2.json"),
                "--out",
                "/runs",
                *plan.campaign_command[1:],
            ]
        elif plan.campaign_command[0].startswith("train-gemma4-e4b"):
            version = plan.campaign_command[0].rsplit("-", 1)[-1]
            if version not in {"v2", "v3"}:
                version = "v1"
            command = [
                "python",
                "/workspace/scripts/train_gemma4_e4b_control.py",
                "--model",
                "/data/gemma4-e4b",
                "--dataset",
                "/workspace/experiments/governed/mars-v2-e4b-v1",
                "--preregistration",
                (f"/workspace/experiments/canonical/gemma4_e4b_governed_control_{version}.json"),
                "--out",
                "/runs",
                *plan.campaign_command[1:],
            ]
        elif plan.campaign_command[0] in {
            "shadow-gemma4-e4b-rezero-v1",
            "shadow-gemma4-e4b-rezero-v2",
            "shadow-gemma4-e4b-rezero-v4",
        }:
            rezero_version = plan.campaign_command[0].rsplit("-", 1)[-1]
            command = [
                "python",
                "/workspace/scripts/run_gemma4_e4b_shadow.py",
                "--model",
                "/data/gemma4-e4b",
                "--bundle",
                f"/data/gemma4-e4b-rezero-control-{rezero_version}-seed17.tar.gz",
                "--cases",
                "/workspace/experiments/shadow/external-v1/cases.jsonl",
                "--preregistration",
                (
                    "/workspace/experiments/canonical/"
                    f"gemma4_e4b_rezero_shadow_external_{rezero_version}.json"
                ),
                "--out",
                "/runs",
                *plan.campaign_command[1:],
            ]
        elif plan.campaign_command[0].startswith("shadow-gemma4-e4b"):
            shadow_version = plan.campaign_command[0].rsplit("-", 1)[-1]
            command = [
                "python",
                "/workspace/scripts/run_gemma4_e4b_shadow.py",
                "--model",
                "/data/gemma4-e4b",
                "--bundle",
                "/data/gemma4-e4b-control-v3-seed17.tar.gz",
                "--cases",
                "/workspace/experiments/shadow/external-v1/cases.jsonl",
                "--preregistration",
                (
                    "/workspace/experiments/canonical/"
                    f"gemma4_e4b_shadow_external_{shadow_version}.json"
                ),
                "--out",
                "/runs",
                *plan.campaign_command[1:],
            ]
        elif plan.campaign_command[0] == "canary-gemma4-e4b-quoted-runtime-v1":
            command = [
                "python",
                "/workspace/scripts/canary_gemma4_e4b_quoted_runtime.py",
                "--model",
                "/data/gemma4-e4b",
                "--bundle",
                "/data/gemma4-e4b-control-v3-seed17.tar.gz",
                "--source-revision",
                plan.revision,
                "--source-patch-sha256",
                plan.campaign_command[2],
                "--out",
                "/runs",
                "--runtime-timeout-seconds",
                "180",
                "--bundle-sha256",
                BUNDLE_SHA256,
            ]
        else:
            if plan.campaign_command[0] == NAVIGATION_UNIFIED_CAMPAIGN:
                command = [
                    "python",
                    "/workspace/scripts/run_hyphae_minilm_gemma_navigation_canary.py",
                    "--hyphae-archive",
                    "/data/hyphae-2.1.0-x86_64-unknown-linux-gnu.tar.gz",
                    "--hyphae-binary",
                    "/data/hyphae-2.1.0",
                    "--hyphae-wheel",
                    "/data/hyphae_sdk-2.1.0-py3-none-any.whl",
                    "--minilm-model",
                    "/data/all-MiniLM-L6-v2",
                    "--gemma-model",
                    "/data/gemma4-e4b",
                    "--pilot",
                    "/data/gemma4-e4b-rezero-navigation-v1-seed17.tar.gz",
                    "--source-revision",
                    plan.revision,
                    "--source-patch-sha256",
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "--work-root",
                    "/tmp/hyphae-navigation-v1",
                    "--out",
                    "/runs",
                ]
                inner = f"cd /workspace && PYTHONPATH=/workspace/src:/python {shlex.join(command)}"
                return " && ".join(
                    (
                        (
                            "timeout --signal=TERM "
                            f"{campaign_seconds}s "
                            f"{_rocm_container_command(plan, network_none=True)} "
                            "/bin/bash -lc " + shlex.quote(inner)
                        ),
                    )
                )
            if plan.campaign_command[0] == NAVIGATION_V2_CAMPAIGN:
                command = [
                    "python",
                    "/workspace/scripts/run_hyphae_minilm_gemma_navigation_canary_v2.py",
                    "--hyphae-archive",
                    "/data/hyphae-2.1.0-x86_64-unknown-linux-gnu.tar.gz",
                    "--hyphae-binary",
                    "/data/hyphae-2.1.0",
                    "--hyphae-wheel",
                    "/data/hyphae_sdk-2.1.0-py3-none-any.whl",
                    "--minilm-model",
                    "/data/all-MiniLM-L6-v2",
                    "--gemma-model",
                    "/data/gemma4-e4b",
                    "--pilot",
                    "/data/gemma4-e4b-rezero-navigation-v2-seed17.tar.gz",
                    "--source-revision",
                    plan.revision,
                    "--source-patch-sha256",
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "--work-root",
                    "/tmp/hyphae-navigation-v2",
                    "--out",
                    "/runs",
                ]
                inner = (
                    f"cd /workspace && PYTHONPATH=/workspace/src:/python {shlex.join(command)}; "
                    "cd /runs && tar -czf - $(ls -A) | base64 -w0"
                )
                return " && ".join(
                    (
                        (
                            "timeout --signal=TERM "
                            f"{campaign_seconds}s "
                            f"{_rocm_container_command(plan, network_none=True)} "
                            "/bin/bash -lc " + shlex.quote(inner)
                        ),
                    )
                )
            if plan.campaign_command[0] == NAVIGATION_V3_CAMPAIGN:
                command = [
                    "python",
                    "/workspace/scripts/run_hyphae_minilm_gemma_navigation_canary_v3.py",
                    "--hyphae-archive",
                    "/data/hyphae-2.1.0-x86_64-unknown-linux-gnu.tar.gz",
                    "--hyphae-binary",
                    "/data/hyphae-2.1.0",
                    "--hyphae-wheel",
                    "/data/hyphae_sdk-2.1.0-py3-none-any.whl",
                    "--minilm-model",
                    "/data/all-MiniLM-L6-v2",
                    "--gemma-model",
                    "/data/gemma4-e4b",
                    "--pilot",
                    "/data/gemma4-e4b-rezero-navigation-v3-seed17.tar.gz",
                    "--source-revision",
                    plan.revision,
                    "--source-patch-sha256",
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "--work-root",
                    "/tmp/hyphae-navigation-v3",
                    "--out",
                    "/runs",
                ]
                inner = (
                    f"cd /workspace && PYTHONPATH=/workspace/src:/python {shlex.join(command)}; "
                    "cd /runs && tar -czf - $(ls -A) | base64 -w0"
                )
                return " && ".join(
                    (
                        (
                            "timeout --signal=TERM "
                            f"{campaign_seconds}s "
                            f"{_rocm_container_command(plan, network_none=True)} "
                            "/bin/bash -lc " + shlex.quote(inner)
                        ),
                    )
                )
            rezero_unified = plan.campaign_command[0] == REZERO_UNIFIED_CAMPAIGN
            command = [
                "python",
                "/workspace/scripts/run_hyphae_minilm_gemma_canary.py",
                "--hyphae-archive",
                "/data/hyphae-2.1.0-x86_64-unknown-linux-gnu.tar.gz",
                "--hyphae-binary",
                "/data/hyphae-2.1.0",
                "--hyphae-wheel",
                "/data/hyphae_sdk-2.1.0-py3-none-any.whl",
                "--minilm-model",
                "/data/all-MiniLM-L6-v2",
                "--gemma-model",
                "/data/gemma4-e4b",
                "--bundle",
                (
                    "/data/gemma4-e4b-rezero-control-v4-seed17.tar.gz"
                    if rezero_unified
                    else "/data/gemma4-e4b-control-v3-seed17.tar.gz"
                ),
                "--source-revision",
                plan.revision,
                "--source-patch-sha256",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "--work-root",
                "/tmp/hyphae-unified-v1",
                "--out",
                "/runs",
            ]
            if rezero_unified:
                command.extend(("--controller-kind", "rezero-v4"))
        inner = f"cd /workspace && PYTHONPATH=/workspace/src:/python {shlex.join(command)}"
        return " && ".join(
            (
                (
                    "timeout --signal=TERM "
                    f"{campaign_seconds}s "
                    f"{_rocm_container_command(plan, network_none=(plan.campaign_command[0] in UNIFIED_CAMPAIGNS))} "  # noqa: E501
                    f"/bin/bash -lc {shlex.quote(inner)}"
                ),
            )
        )
    command = [
        "/root/.local/bin/uv",
        "run",
        "hyphae-transformer",
        *plan.campaign_command,
        "--data-root",
        plan.remote_data_root,
        "--run-root",
        plan.remote_run_root,
    ]
    campaign_seconds = plan.max_lifetime_seconds - CLEANUP_RESERVE_SECONDS
    return " && ".join(
        (
            f"cd {shlex.quote(plan.remote_root)}",
            f"timeout --signal=TERM {campaign_seconds}s {shlex.join(command)}",
        )
    )


def _artifact_command(plan: CloudCampaignPlan, public_ip: str) -> list[str]:
    if plan.accelerator == "amd-rocm":
        if plan.campaign_command[0].startswith("train-gemma4-e4b"):
            artifact = f"tar -C {plan.remote_run_root} -czf - . | base64 -w0"
        elif plan.campaign_command[0].startswith("shadow-gemma4-e4b"):
            artifact = (
                f"tar -C {plan.remote_run_root} -czf - "
                "shadow-report.json shadow-audit.jsonl | base64 -w0"
            )
        elif plan.campaign_command[0] == "canary-gemma4-e4b-quoted-runtime-v1":
            artifact = f"tar -C {plan.remote_run_root} -czf - . | base64 -w0"
        elif plan.campaign_command[0] in UNIFIED_CAMPAIGNS:
            evidence_members = (
                NAVIGATION_EVIDENCE
                if plan.campaign_command[0] == NAVIGATION_UNIFIED_CAMPAIGN
                else UNIFIED_EVIDENCE
            )
            members = " ".join(shlex.quote(value) for value in evidence_members)
            artifact = (
                f"tar --ignore-failed-read -C {plan.remote_run_root} -czf - {members} | base64 -w0"
            )
        elif plan.campaign_command[0] == "smoke-gemma4-e4b-rezero-v1":
            artifact = f"base64 -w0 {plan.remote_run_root}/gemma4-e4b-rezero-smoke.json"
        else:
            artifact = f"base64 -w0 {plan.remote_run_root}/gemma4-e4b-smoke.json"
        return _ssh_command(
            plan,
            public_ip,
            artifact,
        )
    return [
        "rsync",
        "-av",
        "--exclude",
        "checkpoints",
        "-e",
        f"ssh -i {shlex.quote(str(plan.ssh_private_key))} -o StrictHostKeyChecking=accept-new",
        f"root@{public_ip}:{plan.remote_run_root}/",
        f"{plan.artifact_directory}/",
    ]


def _validate_remote_source_patch(
    plan: CloudCampaignPlan,
    public_ip: str,
    runner: CommandRunner,
) -> None:
    if plan.campaign_command[0] not in {
        "canary-gemma4-e4b-quoted-runtime-v1",
        UNIFIED_CAMPAIGN,
        REZERO_UNIFIED_CAMPAIGN,
    }:
        return
    result = runner.run(
        _ssh_command(
            plan,
            public_ip,
            f"git -C {shlex.quote(plan.remote_root)} diff --binary -- . | sha256sum",
        ),
        timeout=60,
    )
    values = result.stdout.split()
    expected = (
        plan.campaign_command[2]
        if plan.campaign_command[0] == "canary-gemma4-e4b-quoted-runtime-v1"
        else "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    if not values or values[0] != expected:
        raise RuntimeError("remote source patch digest does not match the canary plan")


def _expected_artifact_name(plan: CloudCampaignPlan) -> str:
    if plan.accelerator != "amd-rocm":
        raise ValueError("only AMD campaigns declare one exact evidence artifact")
    return (
        "gemma4-e4b-training.tar.gz"
        if plan.campaign_command[0].startswith("train-gemma4-e4b")
        else "gemma4-e4b-shadow.tar.gz"
        if plan.campaign_command[0].startswith("shadow-gemma4-e4b")
        else "gemma4-e4b-quoted-runtime-canary.tar.gz"
        if plan.campaign_command[0] == "canary-gemma4-e4b-quoted-runtime-v1"
        else "hyphae-minilm-gemma-navigation-evidence.tar.gz"
        if plan.campaign_command[0] == NAVIGATION_UNIFIED_CAMPAIGN
        else "hyphae-minilm-gemma-navigation-v3-evidence.tar.gz"
        if plan.campaign_command[0] == NAVIGATION_V3_CAMPAIGN
        else "hyphae-minilm-gemma-navigation-v2-evidence.tar.gz"
        if plan.campaign_command[0] == NAVIGATION_V2_CAMPAIGN
        else "hyphae-minilm-gemma-evidence.tar.gz"
        if plan.campaign_command[0] in UNIFIED_CAMPAIGNS
        else "gemma4-e4b-rezero-smoke.json"
        if plan.campaign_command[0] == "smoke-gemma4-e4b-rezero-v1"
        else "gemma4-e4b-smoke.json"
    )


def _rocm_container_command(plan: CloudCampaignPlan, *, network_none: bool = False) -> str:
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add",
        "video",
        "--ipc=host",
        "--shm-size",
        "16G",
        "-v",
        f"{plan.remote_root}:/workspace",
        "-v",
        f"{plan.remote_data_root}:/data",
        "-v",
        f"{plan.remote_run_root}:/runs",
        "-v",
        f"{plan.remote_data_root}/python:/python",
        ROCM_PYTORCH_IMAGE,
    ]
    if network_none:
        command.insert(4, "--network=none")
    return shlex.join(command)


def _rocm_bootstrap_inner(plan: CloudCampaignPlan) -> str:
    commands = [
        (
            'PYTHONPATH=/python python -c "import torch; assert torch.version.hip; '
            "assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; "
            "assert 'gfx950' in torch.cuda.get_device_properties(0).gcnArchName\""
        ),
        "python -m pip install --target /python transformers==5.14.1",
        (
            "curl -LsSf -o /data/gemma4-e4b-control-v3-seed17.tar.gz "
            "https://github.com/Hyphae-Research-Foundation/hyphae-transformer/"
            "releases/download/governed-control-v3.0.0/"
            "gemma4-e4b-control-v3-seed17.tar.gz"
        ),
        f"echo '{BUNDLE_SHA256}  /data/gemma4-e4b-control-v3-seed17.tar.gz' | sha256sum -c -",
        (
            "PYTHONPATH=/python python /workspace/scripts/download_gemma4_e4b.py "
            "--out /data/gemma4-e4b"
        ),
        (
            "PYTHONPATH=/python python /workspace/scripts/preflight_gemma4_e4b.py "
            "--model /data/gemma4-e4b --require-gpu | tee /runs/gemma4-e4b-preflight.json"
        ),
        "git -C /workspace rev-parse HEAD > /runs/source-revision.txt",
        "PYTHONPATH=/python python -m pip freeze > /runs/python-freeze.txt",
    ]
    if plan.campaign_command[0] in UNIFIED_CAMPAIGNS:
        commands.extend(
            [
                (
                    "echo 'fd6503abbcac18db9a6705682b80a83904389f146e6dd0c4d17fdef49535a5fb  "
                    "/data/hyphae_sdk-2.1.0-py3-none-any.whl' | sha256sum -c -"
                ),
                (
                    "python -m pip install --no-deps --target /python "
                    "/data/hyphae_sdk-2.1.0-py3-none-any.whl"
                ),
                (
                    "curl -LsSf -o /data/hyphae-2.1.0-x86_64-unknown-linux-gnu.tar.gz "
                    "https://github.com/Hyphae-Research-Foundation/hyphae/releases/download/"
                    "v2.1.0/hyphae-2.1.0-x86_64-unknown-linux-gnu.tar.gz"
                ),
                (
                    "echo 'a1e8cf56d9b9a96ee5f230aa4dec92b2541792f7ca4bb40c0dbf761d9ed3e0aa  "
                    "/data/hyphae-2.1.0-x86_64-unknown-linux-gnu.tar.gz' | sha256sum -c -"
                ),
                (
                    "tar -xOf /data/hyphae-2.1.0-x86_64-unknown-linux-gnu.tar.gz "
                    "hyphae-2.1.0-x86_64-unknown-linux-gnu/hyphae > /data/hyphae-2.1.0"
                ),
                "chmod 0500 /data/hyphae-2.1.0",
                (
                    "echo 'a00ea0cfc502ad63d65c42357664f7664f8a8c482fbdeb24a4f5511feceb45d0  "
                    "/data/hyphae-2.1.0' | sha256sum -c -"
                ),
                (
                    "PYTHONPATH=/workspace/src:/python python "
                    "/workspace/scripts/download_minilm_l6_v2.py "
                    "--out /data/all-MiniLM-L6-v2 | tee /runs/minilm-preflight.json"
                ),
            ]
        )
    if plan.campaign_command[0] == REZERO_UNIFIED_CAMPAIGN:
        commands.extend(
            [
                (
                    "curl -LsSf -o /data/gemma4-e4b-rezero-control-v4-seed17.tar.gz "
                    "https://github.com/Hyphae-Research-Foundation/hyphae-transformer/"
                    "releases/download/rezero-control-v4.0.0/"
                    "gemma4-e4b-rezero-control-v4-seed17.tar.gz"
                ),
                (
                    f"echo '{REZERO_V4_BUNDLE_SHA256}  "
                    "/data/gemma4-e4b-rezero-control-v4-seed17.tar.gz' | sha256sum -c -"
                ),
            ]
        )
    if plan.campaign_command[0] == NAVIGATION_UNIFIED_CAMPAIGN:
        commands.extend(
            [
                (
                    "curl -LsSf -o /data/gemma4-e4b-rezero-navigation-v1-seed17.tar.gz "
                    "https://github.com/Hyphae-Research-Foundation/hyphae-transformer/"
                    "releases/download/rezero-navigation-v1.0.0/"
                    "gemma4-e4b-rezero-navigation-v1-seed17.tar.gz"
                ),
                (
                    f"echo '{NAVIGATION_BUNDLE_SHA256}  "
                    "/data/gemma4-e4b-rezero-navigation-v1-seed17.tar.gz' | sha256sum -c -"
                ),
            ]
        )
    if plan.campaign_command[0] == NAVIGATION_V2_CAMPAIGN:
        commands.extend(
            [
                (
                    "curl -LsSf -o /data/gemma4-e4b-rezero-navigation-v2-seed17.tar.gz "
                    "https://github.com/Hyphae-Research-Foundation/hyphae-transformer/"
                    "releases/download/rezero-navigation-v2.0.0/"
                    "gemma4-e4b-rezero-navigation-v2-seed17.tar.gz"
                ),
                (
                    f"echo '{NAVIGATION_V2_BUNDLE_SHA256}  "
                    "/data/gemma4-e4b-rezero-navigation-v2-seed17.tar.gz' | sha256sum -c -"
                ),
            ]
        )
    if plan.campaign_command[0] == NAVIGATION_V3_CAMPAIGN:
        commands.extend(
            [
                (
                    "curl -LsSf -o /data/gemma4-e4b-rezero-navigation-v3-seed17.tar.gz "
                    "https://github.com/Hyphae-Research-Foundation/hyphae-transformer/"
                    "releases/download/rezero-navigation-v3.0.0/"
                    "gemma4-e4b-rezero-navigation-v3-seed17.tar.gz"
                ),
                (
                    f"echo '{NAVIGATION_V3_BUNDLE_SHA256}  "
                    "/data/gemma4-e4b-rezero-navigation-v3-seed17.tar.gz' | sha256sum -c -"
                ),
            ]
        )
    if plan.campaign_command[0] == "shadow-gemma4-e4b-rezero-v1":
        commands.extend(
            [
                (
                    "curl -LsSf -o /data/gemma4-e4b-rezero-control-v1-seed17.tar.gz "
                    "https://github.com/Hyphae-Research-Foundation/hyphae-transformer/"
                    "releases/download/rezero-control-v1.0.0/"
                    "gemma4-e4b-rezero-control-v1-seed17.tar.gz"
                ),
                (
                    f"echo '{REZERO_BUNDLE_SHA256}  "
                    "/data/gemma4-e4b-rezero-control-v1-seed17.tar.gz' | sha256sum -c -"
                ),
            ]
        )
    if plan.campaign_command[0] == "shadow-gemma4-e4b-rezero-v2":
        commands.extend(
            [
                (
                    "curl -LsSf -o /data/gemma4-e4b-rezero-control-v2-seed17.tar.gz "
                    "https://github.com/Hyphae-Research-Foundation/hyphae-transformer/"
                    "releases/download/rezero-control-v2.0.0/"
                    "gemma4-e4b-rezero-control-v2-seed17.tar.gz"
                ),
                (
                    f"echo '{REZERO_V2_BUNDLE_SHA256}  "
                    "/data/gemma4-e4b-rezero-control-v2-seed17.tar.gz' | sha256sum -c -"
                ),
            ]
        )
    if plan.campaign_command[0] == "shadow-gemma4-e4b-rezero-v4":
        commands.extend(
            [
                (
                    "curl -LsSf -o /data/gemma4-e4b-rezero-control-v4-seed17.tar.gz "
                    "https://github.com/Hyphae-Research-Foundation/hyphae-transformer/"
                    "releases/download/rezero-control-v4.0.0/"
                    "gemma4-e4b-rezero-control-v4-seed17.tar.gz"
                ),
                (
                    f"echo '{REZERO_V4_BUNDLE_SHA256}  "
                    "/data/gemma4-e4b-rezero-control-v4-seed17.tar.gz' | sha256sum -c -"
                ),
            ]
        )
    return " && ".join(commands)


def _verify_remote_revision(plan: CloudCampaignPlan) -> None:
    with tempfile.TemporaryDirectory(prefix="hyphae-revision-") as directory:
        subprocess.run(
            ["git", "init", "--bare", "--quiet", directory],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    directory,
                    "fetch",
                    "--quiet",
                    "--depth",
                    "1",
                    plan.repository_url,
                    plan.revision,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(
                "cloud source revision is not fetchable from its repository"
            ) from error


def _write_retrieved_evidence(
    plan: CloudCampaignPlan, process: subprocess.CompletedProcess[str]
) -> None:
    try:
        payload = base64.b64decode(process.stdout, validate=True)
        completed = True
        if plan.campaign_command[0].startswith("train-gemma4-e4b"):
            with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
                report_name = (
                    "./rezero-navigation-report.json"
                    if plan.campaign_command[0] == "train-gemma4-e4b-rezero-navigation-v1"
                    else "./rezero-navigation-v3-report.json"
                    if plan.campaign_command[0] == "train-gemma4-e4b-rezero-navigation-v3"
                    else "./rezero-navigation-v2-report.json"
                    if plan.campaign_command[0] == "train-gemma4-e4b-rezero-navigation-v2"
                    else "./rezero-sequence-report.json"
                    if plan.campaign_command[0].startswith("train-gemma4-e4b-rezero-")
                    else "./training-report.json"
                )
                member = archive.getmember(report_name)
                source = archive.extractfile(member)
                if source is None or not member.isfile():
                    raise ValueError("training report is absent")
                value = json.loads(source.read())
                completed = isinstance(value, dict) and value.get("completed") is True
        elif plan.campaign_command[0].startswith("shadow-gemma4-e4b"):
            with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
                member = archive.getmember("shadow-report.json")
                source = archive.extractfile(member)
                if source is None or not member.isfile():
                    raise ValueError("shadow report is absent")
                value = json.loads(source.read())
                completed = isinstance(value, dict) and value.get("completed") is True
        elif plan.campaign_command[0] == "canary-gemma4-e4b-quoted-runtime-v1":
            with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
                member = archive.getmember("./quoted-runtime-canary-report.json")
                source = archive.extractfile(member)
                if source is None or not member.isfile():
                    raise ValueError("quoted runtime report is absent")
                value = json.loads(source.read())
                completed = (
                    isinstance(value, dict)
                    and value.get("completed") is True
                    and value.get("passed") is True
                    and value.get("request_count") == 1
                )
        elif plan.campaign_command[0] in UNIFIED_CAMPAIGNS:
            value, members = _read_unified_evidence(payload)
            completed = (
                _valid_unified_report(value, plan)
                or (
                    plan.campaign_command[0] == NAVIGATION_UNIFIED_CAMPAIGN
                    and value.get("schema")
                    == "hyphae-transformer.hyphae-minilm-gemma-navigation-canary/v1"
                    and value.get("completed") is True
                )
                or (
                    plan.campaign_command[0] == NAVIGATION_V2_CAMPAIGN
                    and value.get("schema")
                    == "hyphae-transformer.hyphae-minilm-gemma-navigation-canary/v2"
                    and value.get("completed") is True
                )
            )
        else:
            value = json.loads(payload)
            completed = (
                isinstance(value, dict)
                and value.get("passed") is True
                and (
                    plan.campaign_command[0] == "smoke-gemma4-e4b" or value.get("completed") is True
                )
            )
    except (ValueError, KeyError, json.JSONDecodeError, tarfile.TarError) as error:
        raise RuntimeError(f"retrieved cloud campaign evidence is invalid: {error}") from error
    if not completed:
        raise RuntimeError("retrieved cloud campaign evidence did not complete")
    (plan.artifact_directory / _expected_artifact_name(plan)).write_bytes(payload)
    if plan.campaign_command[0].startswith("train-gemma4-e4b"):
        report_name = (
            "rezero-navigation-report.json"
            if plan.campaign_command[0] == "train-gemma4-e4b-rezero-navigation-v1"
            else "rezero-navigation-v3-report.json"
            if plan.campaign_command[0] == "train-gemma4-e4b-rezero-navigation-v3"
            else "rezero-navigation-v2-report.json"
            if plan.campaign_command[0] == "train-gemma4-e4b-rezero-navigation-v2"
            else "rezero-sequence-report.json"
            if plan.campaign_command[0].startswith("train-gemma4-e4b-rezero-")
            else "training-report.json"
        )
        (plan.artifact_directory / report_name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    elif plan.campaign_command[0].startswith("shadow-gemma4-e4b"):
        (plan.artifact_directory / "shadow-report.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    elif plan.campaign_command[0] == "canary-gemma4-e4b-quoted-runtime-v1":
        with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
            archive.extractall(plan.artifact_directory, filter="data")
    elif plan.campaign_command[0] in UNIFIED_CAMPAIGNS:
        for name, content in members.items():
            (plan.artifact_directory / name).write_bytes(content)


def _read_unified_evidence(
    payload: bytes,
) -> tuple[dict[str, object], dict[str, bytes]]:
    if len(payload) > 8_000_000:
        raise ValueError("unified evidence archive is oversized")
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.removeprefix("./")
            if (
                name not in (*UNIFIED_EVIDENCE, *NAVIGATION_EVIDENCE)
                or name in members
                or not member.isfile()
                or member.size > 1_000_000
            ):
                raise ValueError("unified evidence archive member is invalid")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("unified evidence archive member is unreadable")
            content = source.read(1_000_001)
            if len(content) != member.size:
                raise ValueError("unified evidence archive member size differs")
            members[name] = content
    report_name = (
        "navigation-campaign-report.json"
        if "navigation-campaign-report.json" in members
        else "unified-campaign-report.json"
    )
    if set(members) != set(
        NAVIGATION_EVIDENCE
        if report_name == "navigation-campaign-report.json"
        else UNIFIED_EVIDENCE
    ):
        raise ValueError(f"unified evidence archive is incomplete: got {sorted(members)}")
    value = json.loads(members[report_name])
    if not isinstance(value, dict):
        raise ValueError("unified report is not an object")
    return value, members


def _valid_unified_report(value: dict[str, object], plan: CloudCampaignPlan) -> bool:
    if plan.campaign_command[0] in {
        NAVIGATION_UNIFIED_CAMPAIGN,
        NAVIGATION_V2_CAMPAIGN,
        NAVIGATION_V3_CAMPAIGN,
    }:
        return _valid_navigation_report(value, plan)
    dependencies = value.get("dependencies")
    native = value.get("native")
    embedding = value.get("embedding")
    runtime = value.get("runtime")
    finalization = value.get("finalization")
    expected_bundle = (
        REZERO_V4_BUNDLE_SHA256
        if plan.campaign_command[0] == REZERO_UNIFIED_CAMPAIGN
        else BUNDLE_SHA256
    )
    expected_controller = (
        "rezero-v4" if plan.campaign_command[0] == REZERO_UNIFIED_CAMPAIGN else "governed-v3"
    )
    return (
        value.get("schema") == "hyphae-transformer.hyphae-minilm-gemma-canary/v1"
        and value.get("completed") is True
        and value.get("passed") is True
        and value.get("source_revision") == plan.revision
        and isinstance(dependencies, dict)
        and dependencies.get("hyphae_binary_sha256")
        == "a00ea0cfc502ad63d65c42357664f7664f8a8c482fbdeb24a4f5511feceb45d0"
        and dependencies.get("hyphae_wheel_sha256") == HYPHAE_WHEEL_SHA256
        and dependencies.get("minilm_artifact_manifest_sha256")
        == "e5d9d07b6db0c99cc4a2afa92047d57b84c3cb6ed48137ad3612601fdbe21411"
        and dependencies.get("gemma_artifact_manifest_sha256")
        == "662a2f15fe866f2350da4bfeefc746b0eb72c917d344dda2602df88230401561"
        and dependencies.get("bundle_sha256") == expected_bundle
        and isinstance(native, dict)
        and native.get("protocol") == [1, 5]
        and native.get("collection_definition_sha256")
        == ("181552f7f9666546db8f09b3e89be98e99f4c4e09be227f6d257da93029ea527")
        and native.get("durability") == "strict"
        and native.get("strategy") == "exact_filtered"
        and native.get("approximate") is False
        and native.get("exact_reranked") is True
        and native.get("restart_replay") is True
        and isinstance(embedding, dict)
        and embedding.get("dimensions") == 384
        and embedding.get("ready") is True
        and isinstance(runtime, dict)
        and runtime.get("request_count") == 1
        and runtime.get("decision") == "answer"
        and runtime.get("bundle_sha256") == expected_bundle
        and runtime.get("controller_kind") == expected_controller
        and isinstance(finalization, dict)
        and finalization.get("status") == "completed"
        and finalization.get("mailbox_accepted") == 1
    )


def _valid_navigation_report(value: dict[str, object], plan: CloudCampaignPlan) -> bool:
    v2 = plan.campaign_command[0] == NAVIGATION_V2_CAMPAIGN
    v3 = plan.campaign_command[0] == NAVIGATION_V3_CAMPAIGN
    expected_schema = (
        "hyphae-transformer.hyphae-minilm-gemma-navigation-canary/v3"
        if v3
        else "hyphae-transformer.hyphae-minilm-gemma-navigation-canary/v2"
        if v2
        else "hyphae-transformer.hyphae-minilm-gemma-navigation-canary/v1"
    )
    expected_bundle = (
        NAVIGATION_V3_BUNDLE_SHA256
        if v3
        else NAVIGATION_V2_BUNDLE_SHA256
        if v2
        else NAVIGATION_BUNDLE_SHA256
    )
    expected_checkpoint = (
        NAVIGATION_V3_CHECKPOINT_SHA256
        if v3
        else NAVIGATION_V2_CHECKPOINT_SHA256
        if v2
        else "47940ec5f690fab92f13601ca6c1593b8897d062a04c3b853e4fc99fd762aca2"
    )
    dependencies = value.get("dependencies")
    native = value.get("native")
    embedding = value.get("embedding")
    publication = value.get("publication")
    generation = value.get("generation")
    pilot = value.get("pilot")
    steps = pilot.get("steps") if isinstance(pilot, dict) else None
    return (
        value.get("schema") == expected_schema
        and value.get("completed") is True
        and value.get("passed") is True
        and value.get("source_revision") == plan.revision
        and value.get("backbone_unchanged") is True
        and isinstance(dependencies, dict)
        and dependencies.get("hyphae_binary_sha256")
        == "a00ea0cfc502ad63d65c42357664f7664f8a8c482fbdeb24a4f5511feceb45d0"
        and dependencies.get("hyphae_wheel_sha256") == HYPHAE_WHEEL_SHA256
        and dependencies.get("navigation_bundle_sha256") == expected_bundle
        and dependencies.get("navigation_checkpoint_sha256") == expected_checkpoint
        and dependencies.get("minilm_artifact_manifest_sha256")
        == "e5d9d07b6db0c99cc4a2afa92047d57b84c3cb6ed48137ad3612601fdbe21411"
        and dependencies.get("gemma_artifact_manifest_sha256")
        == "662a2f15fe866f2350da4bfeefc746b0eb72c917d344dda2602df88230401561"
        and isinstance(native, dict)
        and native.get("protocol") == [1, 5]
        and native.get("collection_definition_sha256")
        == "181552f7f9666546db8f09b3e89be98e99f4c4e09be227f6d257da93029ea527"
        and native.get("durability") == "strict"
        and native.get("strategy") == "exact_filtered"
        and native.get("approximate") is False
        and native.get("exact_reranked") is True
        and native.get("restart_replay") is True
        and isinstance(embedding, dict)
        and embedding.get("dimensions") == 384
        and embedding.get("ready") is True
        and isinstance(publication, dict)
        and isinstance(generation, dict)
        and isinstance(pilot, dict)
        and pilot.get("maximum_evidence_items") == 8
        and isinstance(steps, list)
        and len(steps) == (3 if v3 else 2)
        and steps[0].get("action") == "search"
        and steps[0].get("selected_handles") == []
        and (steps[1].get("action") == "search" if v3 else steps[1].get("action") == "answer")
        and (
            steps[2].get("action") == "answer"
            and steps[2].get("evidence_handles") == steps[2].get("selected_handles")
            if v3
            else steps[1].get("action") == "answer"
        )
        and (
            steps[2].get("evidence_handles") == steps[2].get("selected_handles")
            if v3
            else len(steps[1].get("evidence_handles", []))
            == len(steps[1].get("selected_handles", [None]))
            and steps[1].get("evidence_handles") == steps[1].get("selected_handles")
        )
    )


def _validate_hyphae_wheel(plan: CloudCampaignPlan) -> None:
    path = plan.hyphae_sdk_wheel
    if path is None or path.is_symlink() or not path.is_file():
        raise ValueError("unified Hyphae SDK wheel is absent or unsafe")
    if path.name != "hyphae_sdk-2.1.0-py3-none-any.whl" or path.stat().st_size != (
        HYPHAE_WHEEL_BYTES
    ):
        raise ValueError("unified Hyphae SDK wheel coordinate differs")
    import hashlib

    if hashlib.sha256(path.read_bytes()).hexdigest() != HYPHAE_WHEEL_SHA256:
        raise ValueError("unified Hyphae SDK wheel digest differs")


def _write_process_evidence(
    plan: CloudCampaignPlan,
    stage: str,
    process: subprocess.CompletedProcess[str],
) -> None:
    plan.artifact_directory.mkdir(parents=True, exist_ok=True)
    (plan.artifact_directory / f"{stage}.stdout.log").write_text(process.stdout)
    (plan.artifact_directory / f"{stage}.stderr.log").write_text(process.stderr)


def _write_failed_process_evidence(
    plan: CloudCampaignPlan,
    stage: str,
    error: subprocess.CalledProcessError,
) -> None:
    plan.artifact_directory.mkdir(parents=True, exist_ok=True)
    stdout = error.stdout if isinstance(error.stdout, str) else ""
    stderr = error.stderr if isinstance(error.stderr, str) else ""
    (plan.artifact_directory / f"{stage}.stdout.log").write_text(stdout)
    (plan.artifact_directory / f"{stage}.stderr.log").write_text(stderr)


def _public_ipv4(droplet: object) -> str:
    if not isinstance(droplet, dict):
        raise TypeError("Droplet response must be an object")
    networks = droplet.get("networks")
    if not isinstance(networks, dict) or not isinstance(networks.get("v4"), list):
        raise RuntimeError("Droplet response has no IPv4 networks")
    for network in networks["v4"]:
        if isinstance(network, dict) and network.get("type") == "public":
            address = network.get("ip_address")
            if isinstance(address, str):
                return address
    raise RuntimeError("Droplet response has no public IPv4 address")


def _delete_and_verify_droplet(runner: CommandRunner, droplet_id: int, *, sleep: object) -> None:
    deletion_error: Exception | None = None
    for attempt in range(3):
        try:
            runner.run(
                ["doctl", "compute", "droplet", "delete", str(droplet_id), "--force"],
                timeout=300,
            )
            deletion_error = None
            break
        except Exception as error:
            deletion_error = error
            if attempt < 2 and callable(sleep):
                sleep(5)

    deadline = time.monotonic() + 60
    last_inventory_error: Exception | None = None
    observed_present = False
    while time.monotonic() < deadline:
        try:
            inventory = runner.run(
                [
                    "doctl",
                    "compute",
                    "droplet",
                    "get",
                    str(droplet_id),
                    "--output",
                    "json",
                ],
                check=False,
                timeout=60,
            )
        except Exception as error:
            last_inventory_error = error
        else:
            if inventory.returncode == 0:
                observed_present = True
            elif _is_droplet_get_404(inventory, droplet_id):
                return
            else:
                last_inventory_error = RuntimeError(
                    "Droplet inventory probe did not return an explicit 404"
                )
        if callable(sleep):
            sleep(min(5, max(0, deadline - time.monotonic())))
    if observed_present:
        raise RuntimeError("deleted Droplet remains in DigitalOcean inventory")
    failure = RuntimeError("Droplet deletion was not confirmed by an explicit 404")
    if last_inventory_error is not None:
        raise failure from last_inventory_error
    if deletion_error is not None:
        raise failure from deletion_error
    raise failure


def _is_droplet_get_404(result: subprocess.CompletedProcess[str], droplet_id: int) -> bool:
    if result.returncode == 0:
        return False
    try:
        value = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return False
    errors = value.get("errors") if isinstance(value, dict) else None
    if not isinstance(errors, list) or len(errors) != 1 or not isinstance(errors[0], dict):
        return False
    detail = errors[0].get("detail")
    return (
        isinstance(detail, str)
        and re.search(rf"^GET \S*/v2/droplets/{droplet_id}: 404(?:\s|$)", detail) is not None
    )


def _validate_gemma_smoke_command(command: tuple[str, ...]) -> None:
    if command != (
        "smoke-gemma4-e4b",
        "--batch-sizes",
        "1",
        "2",
        "4",
        "8",
        "--max-vram-gib",
        "240",
    ):
        raise ValueError("Gemma E4B smoke command differs from preregistration")


def _validate_gemma_rezero_smoke_command(command: tuple[str, ...]) -> None:
    if command != (
        "smoke-gemma4-e4b-rezero-v1",
        "--feature-batch-size",
        "8",
        "--max-vram-gib",
        "240",
    ):
        raise ValueError("Gemma E4B ReZero smoke command differs from preregistration")


def _validate_gemma_training_command(command: tuple[str, ...]) -> None:
    if command != (
        "train-gemma4-e4b",
        "--seeds",
        "17",
        "29",
        "43",
        "--epochs",
        "3",
        "--learning-rate",
        "0.001",
        "--evidence-loss-weight",
        "1.0",
        "--gradient-clip",
        "1.0",
        "--feature-batch-size",
        "8",
    ):
        raise ValueError("Gemma E4B training command differs from preregistration")


def _validate_gemma_v2_training_command(command: tuple[str, ...]) -> None:
    if command != (
        "train-gemma4-e4b-v2",
        "--seeds",
        "17",
        "29",
        "43",
        "--epochs",
        "200",
        "--learning-rate",
        "0.05",
        "--evidence-loss-weight",
        "2.0",
        "--gradient-clip",
        "1.0",
        "--feature-batch-size",
        "8",
        "--pointer-threshold",
        "0.5",
        "--minimum-confidence",
        "0.5",
    ):
        raise ValueError("Gemma E4B v2 training command differs from preregistration")


def _validate_gemma_v3_training_command(command: tuple[str, ...]) -> None:
    if command != (
        "train-gemma4-e4b-v3",
        "--seeds",
        "17",
        "29",
        "43",
        "--epochs",
        "200",
        "--learning-rate",
        "0.05",
        "--evidence-loss-weight",
        "2.0",
        "--gradient-clip",
        "1.0",
        "--feature-batch-size",
        "8",
        "--pointer-threshold",
        "0.5",
        "--minimum-confidence",
        "0.5",
    ):
        raise ValueError("Gemma E4B v3 training command differs from preregistration")


def _validate_gemma_rezero_training_command(command: tuple[str, ...]) -> None:
    if command != (
        "train-gemma4-e4b-rezero-v1",
        "--feature-batch-size",
        "8",
    ):
        raise ValueError("Gemma E4B ReZero training command differs from preregistration")


def _validate_gemma_rezero_v2_training_command(command: tuple[str, ...]) -> None:
    if command != (
        "train-gemma4-e4b-rezero-v2",
        "--feature-batch-size",
        "8",
    ):
        raise ValueError("Gemma E4B ReZero v2 training command differs from preregistration")


def _validate_gemma_rezero_v3_training_command(command: tuple[str, ...]) -> None:
    if command != (
        "train-gemma4-e4b-rezero-v3",
        "--feature-batch-size",
        "8",
    ):
        raise ValueError("Gemma E4B ReZero v3 training command differs from preregistration")


def _validate_gemma_rezero_v4_training_command(command: tuple[str, ...]) -> None:
    if command != (
        "train-gemma4-e4b-rezero-v4",
        "--feature-batch-size",
        "8",
    ):
        raise ValueError("Gemma E4B ReZero v4 training command differs from preregistration")


def _validate_gemma_shadow_command(command: tuple[str, ...]) -> None:
    if command != (
        "shadow-gemma4-e4b-v1",
        "--bundle-sha256",
        "93db742ead71c12fa46c62661b12108fdb0a815d3b5fcf180821538dcfc8b9be",
    ):
        raise ValueError("Gemma E4B shadow command differs from preregistration")


def _validate_gemma_shadow_v2_command(command: tuple[str, ...]) -> None:
    if command != (
        "shadow-gemma4-e4b-v2",
        "--bundle-sha256",
        "93db742ead71c12fa46c62661b12108fdb0a815d3b5fcf180821538dcfc8b9be",
    ):
        raise ValueError("Gemma E4B shadow v2 command differs from preregistration")


def _validate_gemma_rezero_shadow_command(command: tuple[str, ...]) -> None:
    if command != (
        "shadow-gemma4-e4b-rezero-v1",
        "--bundle-sha256",
        REZERO_BUNDLE_SHA256,
    ):
        raise ValueError("Gemma E4B ReZero shadow command differs from preregistration")


def _validate_gemma_rezero_v2_shadow_command(command: tuple[str, ...]) -> None:
    if command != (
        "shadow-gemma4-e4b-rezero-v2",
        "--bundle-sha256",
        REZERO_V2_BUNDLE_SHA256,
    ):
        raise ValueError("Gemma E4B ReZero v2 shadow command differs from preregistration")


def _validate_gemma_rezero_v4_shadow_command(command: tuple[str, ...]) -> None:
    if command != (
        "shadow-gemma4-e4b-rezero-v4",
        "--bundle-sha256",
        REZERO_V4_BUNDLE_SHA256,
    ):
        raise ValueError("Gemma E4B ReZero v4 shadow command differs from preregistration")


def _validate_gemma_quoted_runtime_command(command: tuple[str, ...]) -> None:
    if (
        len(command) != 3
        or command[0] != "canary-gemma4-e4b-quoted-runtime-v1"
        or command[1] != "--source-patch-sha256"
        or not re.fullmatch(r"[0-9a-f]{64}", command[2])
    ):
        raise ValueError("Gemma E4B quoted runtime command differs from preregistration")


def _remaining_lifetime(
    plan: CloudCampaignPlan, created_clock: float, *, reserve_seconds: float = 0
) -> float:
    remaining = plan.max_lifetime_seconds - reserve_seconds - (time.monotonic() - created_clock)
    if remaining <= 0:
        raise TimeoutError("Droplet paid-lifetime budget expired")
    return remaining
