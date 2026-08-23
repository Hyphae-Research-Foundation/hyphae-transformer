"""Fail-safe single-Droplet DigitalOcean campaign executor."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


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
    remote_root: str = "/opt/hyphae-transformer"
    remote_data_root: str = "/opt/celiums-data"
    remote_run_root: str = "/opt/celiums-runs/campaign"

    def __post_init__(self) -> None:
        if not self.name or not self.region or not self.size or not self.image:
            raise ValueError("cloud resource identifiers are required")
        if not self.revision or len(self.revision) < 7:
            raise ValueError("an immutable source revision is required")
        if self.max_lifetime_seconds < 60:
            raise ValueError("cloud lifetime must be at least 60 seconds")
        if min(self.hourly_rate_usd, self.max_cost_usd) <= 0:
            raise ValueError("cloud prices and cost budget must be positive")
        projected_cost = self.hourly_rate_usd * self.max_lifetime_seconds / 3600
        if projected_cost > self.max_cost_usd + 1e-12:
            raise ValueError("maximum lifetime exceeds the cloud cost budget")
        if not self.data_command or self.data_command[0] != "prepare-data":
            raise ValueError("data command must use the prepare-data allowlist")
        if not self.campaign_command or self.campaign_command[0] not in {
            "pilot-wikitext2",
            "pilot-enwiki8",
        }:
            raise ValueError("campaign command is not allowlisted")
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

    droplet_id: int | None = None
    public_ip: str | None = None
    created_at: datetime | None = None
    failure: str | None = None
    status = "failed"
    try:
        created = command_runner.run(commands[0], timeout=600)
        values = json.loads(created.stdout)
        if not isinstance(values, list) or len(values) != 1:
            raise RuntimeError("DigitalOcean create did not return exactly one Droplet")
        droplet = values[0]
        droplet_id = int(droplet["id"])
        public_ip = _public_ipv4(droplet)
        created_at = datetime.now(UTC)
        _wait_for_ssh(plan, public_ip, command_runner, sleep=sleep)
        command_runner.run(_ssh_command(plan, public_ip, _bootstrap_script(plan)), timeout=900)
        command_runner.run(
            _ssh_command(plan, public_ip, _campaign_script(plan)),
            timeout=plan.max_lifetime_seconds + 300,
        )
        plan.artifact_directory.mkdir(parents=True, exist_ok=True)
        command_runner.run(_rsync_command(plan, public_ip), timeout=900)
        status = "completed"
    except Exception as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = f": {error.stderr.strip()}"
        failure = f"{type(error).__name__}: {error}{detail}"
        if public_ip is not None:
            try:
                plan.artifact_directory.mkdir(parents=True, exist_ok=True)
                command_runner.run(
                    _rsync_command(plan, public_ip),
                    check=False,
                    timeout=300,
                )
            except Exception:
                pass
    finally:
        deleted_at = datetime.now(UTC)
        if droplet_id is not None:
            command_runner.run(
                ["doctl", "compute", "droplet", "delete", str(droplet_id), "--force"],
                check=False,
                timeout=300,
            )

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
    return [
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
        ]
    ]


def _wait_for_ssh(
    plan: CloudCampaignPlan,
    public_ip: str,
    runner: CommandRunner,
    *,
    sleep: object,
) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        result = runner.run(
            _ssh_command(plan, public_ip, "true"),
            check=False,
            timeout=20,
        )
        if result.returncode == 0:
            return
        if callable(sleep):
            sleep(10)
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
            plan.repository_url,
            plan.remote_root,
        ]
    )
    return " && ".join(
        (
            f"rm -rf {shlex.quote(plan.remote_root)}",
            checkout,
            f"git -C {shlex.quote(plan.remote_root)} checkout {shlex.quote(plan.revision)}",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            f"cd {shlex.quote(plan.remote_root)}",
            "/root/.local/bin/uv sync --frozen",
            f"mkdir -p {shlex.quote(plan.remote_data_root)} {shlex.quote(plan.remote_run_root)}",
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
    return " && ".join(
        (
            f"cd {shlex.quote(plan.remote_root)}",
            f"timeout --signal=TERM {plan.max_lifetime_seconds}s {shlex.join(command)}",
        )
    )


def _rsync_command(plan: CloudCampaignPlan, public_ip: str) -> list[str]:
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
