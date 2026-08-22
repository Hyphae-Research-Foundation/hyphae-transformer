from __future__ import annotations

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
        revision="0123456789abcdef",
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
    def __init__(self, *, fail_campaign: bool = False, fail_create: bool = False) -> None:
        self.commands: list[list[str]] = []
        self.fail_campaign = fail_campaign
        self.fail_create = fail_create

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
        if command[0] == "ssh" and command[-1] == "true":
            return subprocess.CompletedProcess(command, 0, "", "")
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
    assert runner.commands[-1] == [
        "doctl",
        "compute",
        "droplet",
        "delete",
        "42",
        "--force",
    ]
    assert (tmp_path / "artifacts" / "cloud-execution.json").exists()


def test_cloud_executor_retrieves_and_deletes_after_failure(tmp_path: Path) -> None:
    runner = FakeRunner(fail_campaign=True)
    summary = execute_digitalocean_campaign(plan(tmp_path), runner=runner, sleep=lambda _: None)
    assert summary.status == "failed"
    assert summary.failure is not None
    assert any(command[0] == "rsync" for command in runner.commands)
    assert runner.commands[-1][-2:] == ["42", "--force"]


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
