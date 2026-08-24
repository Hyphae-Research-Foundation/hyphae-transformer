from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

from celiums_rezero.cloud.digitalocean import (
    CloudCampaignPlan,
    execute_digitalocean_campaign,
)


def plan(tmp_path: Path) -> CloudCampaignPlan:
    return CloudCampaignPlan(
        name="celiums-rezero-test",
        region="tor1",
        size="gpu-4000adax1-20gb",
        image="gpu-h100x1-base",
        ssh_key_id="1",
        ssh_private_key=tmp_path / "key",
        repository_url="https://github.com/celiumsai/celiums-rezero.git",
        revision="0123456789abcdef0123456789abcdef01234567",
        data_command=("prepare-data", "wikitext2"),
        campaign_command=("pilot-wikitext2", "--device", "cuda"),
        artifact_directory=tmp_path / "artifacts",
        hourly_rate_usd=0.76,
        max_lifetime_seconds=3600,
        max_cost_usd=0.76,
    )


def test_cloud_dry_run_has_no_side_effects(tmp_path: Path) -> None:
    summary = execute_digitalocean_campaign(plan(tmp_path), dry_run=True)
    assert summary.status == "dry_run"
    assert summary.droplet_id is None
    assert summary.dry_run_commands[0][:4] == (
        "doctl",
        "compute",
        "droplet",
        "create",
    )
    assert not (tmp_path / "artifacts").exists()


class FakeRunner:
    def __init__(
        self,
        *,
        fail_campaign: bool = False,
        fail_create: bool = False,
        fail_delete: bool = False,
    ) -> None:
        self.commands: list[list[str]] = []
        self.fail_campaign = fail_campaign
        self.fail_create = fail_create
        self.fail_delete = fail_delete

    def run(
        self,
        command: list[str],
        *,
        capture_output: bool = True,
        check: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, timeout
        self.commands.append(command)
        if command[:4] == ["doctl", "compute", "droplet", "create"]:
            if self.fail_create:
                raise subprocess.CalledProcessError(
                    1,
                    command,
                    stderr="Size is not available in this region.",
                )
            payload = [
                {
                    "id": 42,
                    "networks": {
                        "v4": [{"type": "public", "ip_address": "192.0.2.10"}]
                    },
                }
            ]
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if command[0] == "ssh" and command[-1] == "printf __HYPHAE_READY__":
            return subprocess.CompletedProcess(command, 0, "__HYPHAE_READY__", "")
        if command[0] == "ssh" and command[-1].startswith("base64 -w0 "):
            payload = base64.b64encode(json.dumps({"passed": True}).encode()).decode()
            return subprocess.CompletedProcess(command, 0, payload, "")
        if command[0] == "ssh" and command[-1].startswith("tar -C "):
            import io
            import tarfile

            report = json.dumps({"completed": True, "passed": False}).encode()
            archive_bytes = io.BytesIO()
            with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
                item = tarfile.TarInfo("./training-report.json")
                item.size = len(report)
                archive.addfile(item, io.BytesIO(report))
            payload = base64.b64encode(archive_bytes.getvalue()).decode()
            return subprocess.CompletedProcess(command, 0, payload, "")
        if self.fail_delete and command[:4] == ["doctl", "compute", "droplet", "delete"]:
            raise subprocess.CalledProcessError(1, command, stderr="deletion failed")
        if command[:4] == ["doctl", "compute", "droplet", "get"]:
            return subprocess.CompletedProcess(command, 1, "", "not found")
        if self.fail_campaign and command[0] == "ssh" and "timeout --signal" in command[-1]:
            error = subprocess.CalledProcessError(1, command, stderr="campaign failed")
            if check:
                raise error
            return subprocess.CompletedProcess(command, 1, "", "campaign failed")
        return subprocess.CompletedProcess(command, 0, "", "")


def test_cloud_executor_always_deletes_after_success(tmp_path: Path) -> None:
    runner = FakeRunner()
    summary = execute_digitalocean_campaign(plan(tmp_path), runner=runner, sleep=lambda _: None)
    assert summary.status == "completed"
    assert summary.droplet_id == 42
    assert runner.commands[-2] == [
        "doctl",
        "compute",
        "droplet",
        "delete",
        "42",
        "--force",
    ]
    assert runner.commands[-1][:4] == ["doctl", "compute", "droplet", "get"]
    assert (tmp_path / "artifacts" / "cloud-execution.json").exists()


def test_cloud_executor_retrieves_and_deletes_after_failure(tmp_path: Path) -> None:
    runner = FakeRunner(fail_campaign=True)
    summary = execute_digitalocean_campaign(plan(tmp_path), runner=runner, sleep=lambda _: None)
    assert summary.status == "failed"
    assert summary.failure is not None
    assert any(command[0] == "rsync" for command in runner.commands)
    assert runner.commands[-2][-2:] == ["42", "--force"]
    assert runner.commands[-1][:4] == ["doctl", "compute", "droplet", "get"]


def test_cloud_executor_preserves_create_failure_detail_without_delete(tmp_path: Path) -> None:
    runner = FakeRunner(fail_create=True)
    summary = execute_digitalocean_campaign(plan(tmp_path), runner=runner)
    assert summary.status == "failed"
    assert summary.droplet_id is None
    assert summary.estimated_cost_usd == 0
    assert summary.failure is not None and "Size is not available" in summary.failure
    assert not any(
        command[:4] == ["doctl", "compute", "droplet", "delete"]
        for command in runner.commands
    )


def test_campaign_timeout_allows_cleanup_grace(tmp_path: Path) -> None:
    runner = FakeRunner()
    cloud_plan = plan(tmp_path)
    execute_digitalocean_campaign(cloud_plan, runner=runner, sleep=lambda _: None)
    campaign_calls = [
        command
        for command in runner.commands
        if command[0] == "ssh" and "timeout --signal" in command[-1]
    ]
    assert campaign_calls


def test_cloud_executor_reports_deletion_failure(tmp_path: Path) -> None:
    runner = FakeRunner(fail_delete=True)
    summary = execute_digitalocean_campaign(plan(tmp_path), runner=runner, sleep=lambda _: None)
    assert summary.status == "failed"
    assert summary.failure is not None and "DropletDeletionError" in summary.failure
    delete_attempts = [
        command
        for command in runner.commands
        if command[:4] == ["doctl", "compute", "droplet", "delete"]
    ]
    assert len(delete_attempts) == 3


def test_gemma_rocm_plan_is_strictly_allowlisted(tmp_path: Path) -> None:
    cloud_plan = CloudCampaignPlan(
        name="hyphae-e4b-control-smoke-x1",
        region="mem1",
        size="gpu-mi355x1-288gb-spot",
        image="amddevelopercloud-pytorch2100rocm724",
        ssh_key_id="1",
        ssh_private_key=tmp_path / "key",
        repository_url=(
            "https://github.com/Hyphae-Research-Foundation/hyphae-transformer.git"
        ),
        revision="0123456789abcdef0123456789abcdef01234567",
        data_command=("prepare-gemma4-e4b",),
        campaign_command=(
            "smoke-gemma4-e4b",
            "--batch-sizes",
            "1",
            "2",
            "4",
            "8",
            "--max-vram-gib",
            "240",
        ),
        artifact_directory=tmp_path / "artifacts",
        hourly_rate_usd=4.5,
        max_lifetime_seconds=28_800,
        max_cost_usd=36,
        accelerator="amd-rocm",
    )
    summary = execute_digitalocean_campaign(cloud_plan, dry_run=True)
    create = summary.dry_run_commands[0]
    assert "gpu-mi355x1-288gb-spot" in create


def test_gemma_rocm_executor_retrieves_only_smoke_evidence(tmp_path: Path) -> None:
    cloud_plan = gemma_plan(tmp_path)
    runner = FakeRunner()
    summary = execute_digitalocean_campaign(
        cloud_plan, runner=runner, sleep=lambda _: None
    )
    assert summary.status == "completed"
    retrieval = next(
        command for command in runner.commands if command[-1].startswith("base64 -w0 ")
    )
    assert retrieval[-1].endswith("/gemma4-e4b-smoke.json")
    assert not any(command[0] == "rsync" for command in runner.commands)
    assert (cloud_plan.artifact_directory / "gemma4-e4b-smoke.json").is_file()
    assert (cloud_plan.artifact_directory / "bootstrap.stdout.log").is_file()
    assert (cloud_plan.artifact_directory / "campaign.stdout.log").is_file()


def test_gemma_rocm_bootstrap_uses_pinned_amd_container(tmp_path: Path) -> None:
    runner = FakeRunner()
    execute_digitalocean_campaign(gemma_plan(tmp_path), runner=runner, sleep=lambda _: None)
    bootstrap = next(
        command[-1]
        for command in runner.commands
        if command[0] == "ssh" and "git clone" in command[-1]
    )
    assert "rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1" in bootstrap
    assert "--device=/dev/kfd" in bootstrap
    assert "/workspace/scripts/download_gemma4_e4b.py" in bootstrap
    assert "assert torch.version.hip" in bootstrap
    assert "gcnArchName" in bootstrap


def test_gemma_rocm_campaign_sets_source_pythonpath(tmp_path: Path) -> None:
    runner = FakeRunner()
    execute_digitalocean_campaign(gemma_plan(tmp_path), runner=runner, sleep=lambda _: None)
    campaign = next(
        command[-1]
        for command in runner.commands
        if command[0] == "ssh" and "smoke_gemma4_e4b.py" in command[-1]
    )
    assert "PYTHONPATH=/workspace/src:/python" in campaign


def test_gemma_rocm_bootstrap_persists_python_dependencies(tmp_path: Path) -> None:
    runner = FakeRunner()
    execute_digitalocean_campaign(gemma_plan(tmp_path), runner=runner, sleep=lambda _: None)
    bootstrap = next(
        command[-1]
        for command in runner.commands
        if command[0] == "ssh" and "git clone" in command[-1]
    )
    assert "pip install --target /python transformers==5.14.1" in bootstrap
    assert "/opt/celiums-data/python:/python" in bootstrap


def test_gemma_training_plan_is_strict_and_retrieves_archive(tmp_path: Path) -> None:
    cloud_plan = gemma_training_plan(tmp_path)
    runner = FakeRunner()
    summary = execute_digitalocean_campaign(
        cloud_plan, runner=runner, sleep=lambda _: None
    )
    assert summary.status == "completed"
    assert (cloud_plan.artifact_directory / "gemma4-e4b-training.tar.gz").is_file()
    report = json.loads(
        (cloud_plan.artifact_directory / "training-report.json").read_text()
    )
    assert report == {"completed": True, "passed": False}
    campaign = next(
        command[-1]
        for command in runner.commands
        if command[0] == "ssh" and "train_gemma4_e4b_control.py" in command[-1]
    )
    assert "--seeds 17 29 43" in campaign
    assert "--feature-batch-size 8" in campaign


def test_gemma_v2_training_plan_is_strict(tmp_path: Path) -> None:
    cloud_plan = gemma_v2_training_plan(tmp_path)
    runner = FakeRunner()
    summary = execute_digitalocean_campaign(
        cloud_plan, runner=runner, sleep=lambda _: None
    )
    assert summary.status == "completed"
    campaign = next(
        command[-1]
        for command in runner.commands
        if command[0] == "ssh" and "train_gemma4_e4b_control.py" in command[-1]
    )
    assert "gemma4_e4b_governed_control_v2.json" in campaign
    assert "--epochs 200" in campaign
    assert "--minimum-confidence 0.5" in campaign


def test_gemma_v3_training_plan_is_strict(tmp_path: Path) -> None:
    cloud_plan = gemma_v3_training_plan(tmp_path)
    runner = FakeRunner()
    summary = execute_digitalocean_campaign(
        cloud_plan, runner=runner, sleep=lambda _: None
    )
    assert summary.status == "completed"
    campaign = next(
        command[-1]
        for command in runner.commands
        if command[0] == "ssh" and "train_gemma4_e4b_control.py" in command[-1]
    )
    assert "gemma4_e4b_governed_control_v3.json" in campaign
    assert "--epochs 200" in campaign


def test_cloud_plan_requires_full_commit_sha(tmp_path: Path) -> None:
    values = {
        field: getattr(plan(tmp_path), field)
        for field in CloudCampaignPlan.__dataclass_fields__
        if field not in {"revision"}
    }
    try:
        CloudCampaignPlan(**values, revision="0123456")
    except ValueError as error:
        assert "immutable source revision" in str(error)
    else:
        raise AssertionError("short cloud source revision was accepted")


def gemma_plan(tmp_path: Path) -> CloudCampaignPlan:
    return CloudCampaignPlan(
        name="hyphae-e4b-control-smoke-x1",
        region="mem1",
        size="gpu-mi355x1-288gb-spot",
        image="amddevelopercloud-pytorch2100rocm724",
        ssh_key_id="1",
        ssh_private_key=tmp_path / "key",
        repository_url=(
            "https://github.com/Hyphae-Research-Foundation/hyphae-transformer.git"
        ),
        revision="0123456789abcdef0123456789abcdef01234567",
        data_command=("prepare-gemma4-e4b",),
        campaign_command=(
            "smoke-gemma4-e4b",
            "--batch-sizes",
            "1",
            "2",
            "4",
            "8",
            "--max-vram-gib",
            "240",
        ),
        artifact_directory=tmp_path / "artifacts",
        hourly_rate_usd=4.5,
        max_lifetime_seconds=28_800,
        max_cost_usd=36,
        accelerator="amd-rocm",
    )


def gemma_training_plan(tmp_path: Path) -> CloudCampaignPlan:
    return CloudCampaignPlan(
        name="hyphae-e4b-control-train-x1",
        region="mem1",
        size="gpu-mi355x1-288gb-spot",
        image="amddevelopercloud-pytorch2100rocm724",
        ssh_key_id="1",
        ssh_private_key=tmp_path / "key",
        repository_url=(
            "https://github.com/Hyphae-Research-Foundation/hyphae-transformer.git"
        ),
        revision="0123456789abcdef0123456789abcdef01234567",
        data_command=("prepare-gemma4-e4b",),
        campaign_command=(
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
        ),
        artifact_directory=tmp_path / "artifacts",
        hourly_rate_usd=4.5,
        max_lifetime_seconds=28_800,
        max_cost_usd=36,
        accelerator="amd-rocm",
    )


def gemma_v2_training_plan(tmp_path: Path) -> CloudCampaignPlan:
    return CloudCampaignPlan(
        name="hyphae-e4b-control-train-v2-x1",
        region="mem1",
        size="gpu-mi355x1-288gb-spot",
        image="amddevelopercloud-pytorch2100rocm724",
        ssh_key_id="1",
        ssh_private_key=tmp_path / "key",
        repository_url=(
            "https://github.com/Hyphae-Research-Foundation/hyphae-transformer.git"
        ),
        revision="0123456789abcdef0123456789abcdef01234567",
        data_command=("prepare-gemma4-e4b",),
        campaign_command=(
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
        ),
        artifact_directory=tmp_path / "artifacts",
        hourly_rate_usd=4.5,
        max_lifetime_seconds=28_800,
        max_cost_usd=36,
        accelerator="amd-rocm",
    )


def gemma_v3_training_plan(tmp_path: Path) -> CloudCampaignPlan:
    values = {
        field: getattr(gemma_v2_training_plan(tmp_path), field)
        for field in CloudCampaignPlan.__dataclass_fields__
    }
    values.update(
        name="hyphae-e4b-control-train-v3-x1",
        campaign_command=(
            "train-gemma4-e4b-v3",
            *values["campaign_command"][1:],
        ),
    )
    return CloudCampaignPlan(**values)
