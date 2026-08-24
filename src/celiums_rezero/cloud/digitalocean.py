"""Fail-safe single-Droplet DigitalOcean campaign executor."""

from __future__ import annotations

import base64
import json
import re
import shlex
import subprocess
import tarfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Protocol

CLEANUP_RESERVE_SECONDS = 300
ROCM_PYTORCH_IMAGE = "rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1"


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
            "train-gemma4-e4b",
            "train-gemma4-e4b-v2",
            "train-gemma4-e4b-v3",
            "shadow-gemma4-e4b-v1",
        }:
            raise ValueError("campaign command is not allowlisted")
        if self.accelerator not in {"nvidia", "amd-rocm"}:
            raise ValueError("cloud accelerator is not allowlisted")
        gemma_workload = self.data_command[0] == "prepare-gemma4-e4b" or (
            self.campaign_command[0]
            in {
                "smoke-gemma4-e4b",
                "train-gemma4-e4b",
                "train-gemma4-e4b-v2",
                "train-gemma4-e4b-v3",
                "shadow-gemma4-e4b-v1",
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
        if self.campaign_command[0] == "train-gemma4-e4b":
            _validate_gemma_training_command(self.campaign_command)
        if self.campaign_command[0] == "train-gemma4-e4b-v2":
            _validate_gemma_v2_training_command(self.campaign_command)
        if self.campaign_command[0] == "train-gemma4-e4b-v3":
            _validate_gemma_v3_training_command(self.campaign_command)
        if self.campaign_command[0] == "shadow-gemma4-e4b-v1":
            _validate_gemma_shadow_command(self.campaign_command)
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
    created_clock: float | None = None
    failure: str | None = None
    status = "failed"
    try:
        create_started_at = datetime.now(UTC)
        create_started_clock = time.monotonic()
        created = command_runner.run(
            commands[0], timeout=min(600, plan.max_lifetime_seconds)
        )
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
                _remaining_lifetime(
                    plan, created_clock, reserve_seconds=CLEANUP_RESERVE_SECONDS
                ),
            ),
            sleep=sleep,
        )
        bootstrap = command_runner.run(
            _ssh_command(plan, public_ip, _bootstrap_script(plan)),
            timeout=_remaining_lifetime(
                plan, created_clock, reserve_seconds=CLEANUP_RESERVE_SECONDS
            ),
        )
        _write_process_evidence(plan, "bootstrap", bootstrap)
        campaign = command_runner.run(
            _ssh_command(plan, public_ip, _campaign_script(plan)),
            timeout=_remaining_lifetime(
                plan, created_clock, reserve_seconds=CLEANUP_RESERVE_SECONDS
            ),
        )
        _write_process_evidence(plan, "campaign", campaign)
        plan.artifact_directory.mkdir(parents=True, exist_ok=True)
        retrieval = command_runner.run(
            _artifact_command(plan, public_ip),
            timeout=min(
                900,
                _remaining_lifetime(
                    plan, created_clock, reserve_seconds=CLEANUP_RESERVE_SECONDS
                ),
            ),
        )
        if plan.accelerator == "amd-rocm":
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
                failure = (
                    deletion_failure if failure is None else f"{failure}; {deletion_failure}"
                )
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
            plan.repository_url,
            plan.remote_root,
        ]
    )
    common = (
        f"rm -rf {shlex.quote(plan.remote_root)}",
        checkout,
        f"git -C {shlex.quote(plan.remote_root)} checkout {shlex.quote(plan.revision)}",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"mkdir -p {shlex.quote(plan.remote_data_root)} {shlex.quote(plan.remote_run_root)}",
    )
    if plan.accelerator == "amd-rocm":
        container = _rocm_container_command(plan)
        return " && ".join(
            (
                *common,
                "rocm-smi --showproductname --showmeminfo vram --showdriverversion",
                f"docker pull {shlex.quote(ROCM_PYTORCH_IMAGE)}",
                f"{container} /bin/bash -lc {shlex.quote(_rocm_bootstrap_inner())}",
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
        "train-gemma4-e4b",
        "train-gemma4-e4b-v2",
        "train-gemma4-e4b-v3",
        "shadow-gemma4-e4b-v1",
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
                (
                    "/workspace/experiments/canonical/"
                    f"gemma4_e4b_governed_control_{version}.json"
                ),
                "--out",
                "/runs",
                *plan.campaign_command[1:],
            ]
        else:
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
                "/workspace/experiments/canonical/gemma4_e4b_shadow_external_v1.json",
                "--out",
                "/runs",
                *plan.campaign_command[1:],
            ]
        inner = (
            f"cd /workspace && PYTHONPATH=/workspace/src:/python {shlex.join(command)}"
        )
        return " && ".join(
            (
                (
                    "timeout --signal=TERM "
                    f"{campaign_seconds}s "
                    f"{_rocm_container_command(plan)} /bin/bash -lc {shlex.quote(inner)}"
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
        elif plan.campaign_command[0] == "shadow-gemma4-e4b-v1":
            artifact = (
                f"tar -C {plan.remote_run_root} -czf - "
                "shadow-report.json shadow-audit.jsonl | base64 -w0"
            )
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


def _expected_artifact_name(plan: CloudCampaignPlan) -> str:
    if plan.accelerator != "amd-rocm":
        raise ValueError("only AMD campaigns declare one exact evidence artifact")
    return (
        "gemma4-e4b-training.tar.gz"
        if plan.campaign_command[0].startswith("train-gemma4-e4b")
        else "gemma4-e4b-shadow.tar.gz"
        if plan.campaign_command[0] == "shadow-gemma4-e4b-v1"
        else "gemma4-e4b-smoke.json"
    )


def _rocm_container_command(plan: CloudCampaignPlan) -> str:
    return shlex.join(
        [
            "docker",
            "run",
            "--rm",
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
    )


def _rocm_bootstrap_inner() -> str:
    return " && ".join(
        (
            (
                "PYTHONPATH=/python python -c \"import torch; assert torch.version.hip; "
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
            (
                "PYTHONPATH=/python python /workspace/scripts/download_gemma4_e4b.py "
                "--out /data/gemma4-e4b"
            ),
            (
                "PYTHONPATH=/python python /workspace/scripts/preflight_gemma4_e4b.py "
                "--model /data/gemma4-e4b --require-gpu"
            ),
        )
    )


def _write_retrieved_evidence(
    plan: CloudCampaignPlan, process: subprocess.CompletedProcess[str]
) -> None:
    try:
        payload = base64.b64decode(process.stdout, validate=True)
        completed = True
        if plan.campaign_command[0].startswith("train-gemma4-e4b"):
            with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
                member = archive.getmember("./training-report.json")
                source = archive.extractfile(member)
                if source is None or not member.isfile():
                    raise ValueError("training report is absent")
                value = json.loads(source.read())
                completed = isinstance(value, dict) and value.get("completed") is True
        elif plan.campaign_command[0] == "shadow-gemma4-e4b-v1":
            with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
                member = archive.getmember("shadow-report.json")
                source = archive.extractfile(member)
                if source is None or not member.isfile():
                    raise ValueError("shadow report is absent")
                value = json.loads(source.read())
                completed = (
                    isinstance(value, dict)
                    and value.get("completed") is True
                )
        else:
            value = json.loads(payload)
            completed = isinstance(value, dict) and value.get("passed") is True
    except (ValueError, KeyError, json.JSONDecodeError, tarfile.TarError) as error:
        raise RuntimeError("retrieved cloud campaign evidence is invalid") from error
    if not completed:
        raise RuntimeError("retrieved cloud campaign evidence did not complete")
    (plan.artifact_directory / _expected_artifact_name(plan)).write_bytes(payload)
    if plan.campaign_command[0].startswith("train-gemma4-e4b"):
        (plan.artifact_directory / "training-report.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    elif plan.campaign_command[0] == "shadow-gemma4-e4b-v1":
        (plan.artifact_directory / "shadow-report.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )


def _write_process_evidence(
    plan: CloudCampaignPlan,
    stage: str,
    process: subprocess.CompletedProcess[str],
) -> None:
    plan.artifact_directory.mkdir(parents=True, exist_ok=True)
    (plan.artifact_directory / f"{stage}.stdout.log").write_text(process.stdout)
    (plan.artifact_directory / f"{stage}.stderr.log").write_text(process.stderr)


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


def _delete_and_verify_droplet(
    runner: CommandRunner, droplet_id: int, *, sleep: object
) -> None:
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
    if deletion_error is not None:
        raise deletion_error

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
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
        if inventory.returncode != 0:
            return
        if callable(sleep):
            sleep(min(5, max(0, deadline - time.monotonic())))
    raise RuntimeError("deleted Droplet remains in DigitalOcean inventory")


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


def _validate_gemma_shadow_command(command: tuple[str, ...]) -> None:
    if command != (
        "shadow-gemma4-e4b-v1",
        "--bundle-sha256",
        "93db742ead71c12fa46c62661b12108fdb0a815d3b5fcf180821538dcfc8b9be",
    ):
        raise ValueError("Gemma E4B shadow command differs from preregistration")


def _remaining_lifetime(
    plan: CloudCampaignPlan, created_clock: float, *, reserve_seconds: float = 0
) -> float:
    remaining = (
        plan.max_lifetime_seconds
        - reserve_seconds
        - (time.monotonic() - created_clock)
    )
    if remaining <= 0:
        raise TimeoutError("Droplet paid-lifetime budget expired")
    return remaining
