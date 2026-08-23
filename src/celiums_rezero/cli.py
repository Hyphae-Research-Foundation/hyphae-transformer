"""Command-line entry point for Core smokes and staged Lab runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from celiums_rezero.cloud.digitalocean import CloudCampaignPlan, execute_digitalocean_campaign
from celiums_rezero.core.diagnostics import collect_gate_stats
from celiums_rezero.data.bytes import ByteTokenizer
from celiums_rezero.data.prepare import (
    ENWIKI8_SHA256,
    ENWIKI8_SPLITS,
    WIKITEXT2_REVISION,
    enwiki8_path,
    prepare_public_corpus,
    wikitext2_paths,
)
from celiums_rezero.knowledge.schemas import SufficiencyPolicy
from celiums_rezero.lab.campaign import render_campaign_report, summarize_campaign
from celiums_rezero.lab.registry import Registry
from celiums_rezero.lab.runner import (
    run_manifest,
    run_registered_corpus,
    run_registered_synthetic,
    staged_config,
    staged_corpus_config,
)
from celiums_rezero.lab.schemas import Budget, Hypothesis, RunManifest, RunStage
from celiums_rezero.lab.serialization import to_primitive
from celiums_rezero.training.trainer import TrainConfig
from celiums_rezero.transformer.config import ModelConfig, ResidualStrategy
from celiums_rezero.transformer.model import ReZeroLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyphae-transformer")
    commands = parser.add_subparsers(dest="command", required=True)

    smoke_model = commands.add_parser("smoke-model", help="run a model forward/backward smoke")
    smoke_model.add_argument(
        "--strategy", choices=[item.value for item in ResidualStrategy], default="crz_rms"
    )
    smoke_model.add_argument("--device", default="auto")
    smoke_model.add_argument("--layers", type=int, default=4)

    smoke_lab = commands.add_parser("smoke-lab", help="register and execute a staged run")
    smoke_lab.add_argument("--root", type=Path, default=Path("runs/smoke"))
    smoke_lab.add_argument("--strategy", default="crz_rms")
    smoke_lab.add_argument("--minimum-effect", type=float, default=0.0)
    smoke_lab.add_argument("--device", default="auto")
    smoke_lab.add_argument("--steps", type=int, default=5)

    prepare = commands.add_parser("prepare-data", help="download a public research corpus")
    prepare.add_argument("name", choices=["enwiki8", "wikitext2"])
    prepare.add_argument("--root", type=Path, default=Path("data"))

    execute = commands.add_parser("run-manifest", help="execute an immutable run manifest")
    execute.add_argument("manifest", type=Path)
    execute.add_argument("--registry", type=Path, required=True)
    execute.add_argument("--data-root", type=Path)
    execute.add_argument("--governed-dataset-manifest", type=Path)

    pilot = commands.add_parser("pilot-wikitext2", help="run a WikiText-2 campaign")
    pilot.add_argument("--data-root", type=Path, default=Path("data"))
    pilot.add_argument("--run-root", type=Path, default=Path("runs/wikitext2-mini"))
    pilot.add_argument("--device", default="auto")
    pilot.add_argument("--steps", type=int, default=25)
    pilot.add_argument("--batch-size", type=int, default=4)
    pilot.add_argument("--sequence-length", type=int, default=128)
    pilot.add_argument("--layers", type=int, default=4)
    pilot.add_argument("--d-model", type=int, default=128)
    pilot.add_argument("--train-bytes", type=int, default=1_000_000)
    pilot.add_argument("--evaluation-bytes", type=int, default=100_000)
    pilot.add_argument("--seeds", nargs="+", type=int, default=[7])
    pilot.add_argument("--minimum-effect", type=float, default=0.01)
    pilot.add_argument("--max-wall-seconds", type=float, default=1800)
    pilot.add_argument("--max-device-hours", type=float, default=0.5)
    pilot.add_argument("--max-artifact-bytes", type=int, default=100_000_000)
    pilot.add_argument("--max-failures", type=int, default=0)
    pilot.add_argument("--validation-every-steps", type=int)
    pilot.add_argument("--validation-nll-threshold", type=float)
    pilot.add_argument(
        "--baseline", choices=[item.value for item in ResidualStrategy], default="pre_rms"
    )
    pilot.add_argument(
        "--candidate", choices=[item.value for item in ResidualStrategy], default="crz_rms"
    )
    pilot.add_argument(
        "--stage",
        choices=[RunStage.MINI_PILOT.value, RunStage.PILOT.value],
        default=RunStage.PILOT.value,
    )
    pilot.add_argument(
        "--strategies",
        nargs="+",
        choices=[item.value for item in ResidualStrategy],
        default=[item.value for item in ResidualStrategy],
    )

    enwiki8 = commands.add_parser("pilot-enwiki8", help="run a raw-byte enwiki8 campaign")
    enwiki8.add_argument("--data-root", type=Path, default=Path("data"))
    enwiki8.add_argument("--run-root", type=Path, default=Path("runs/enwiki8-pilot"))
    enwiki8.add_argument("--device", default="auto")
    enwiki8.add_argument("--steps", type=int, default=100)
    enwiki8.add_argument("--batch-size", type=int, default=8)
    enwiki8.add_argument("--sequence-length", type=int, default=256)
    enwiki8.add_argument("--layers", type=int, default=8)
    enwiki8.add_argument("--d-model", type=int, default=256)
    enwiki8.add_argument("--train-bytes", type=int, default=1_000_000)
    enwiki8.add_argument("--evaluation-bytes", type=int, default=100_000)
    enwiki8.add_argument("--seeds", nargs="+", type=int, default=[7])
    enwiki8.add_argument("--minimum-effect", type=float, default=0.01)
    enwiki8.add_argument("--max-wall-seconds", type=float, default=3600)
    enwiki8.add_argument("--max-device-hours", type=float, default=1.0)
    enwiki8.add_argument("--max-artifact-bytes", type=int, default=100_000_000)
    enwiki8.add_argument("--max-failures", type=int, default=0)
    enwiki8.add_argument("--validation-every-steps", type=int)
    enwiki8.add_argument("--validation-nll-threshold", type=float)
    enwiki8.add_argument(
        "--baseline", choices=[item.value for item in ResidualStrategy], default="pre_rms"
    )
    enwiki8.add_argument(
        "--candidate", choices=[item.value for item in ResidualStrategy], default="crz_rms"
    )
    enwiki8.add_argument(
        "--strategies",
        nargs="+",
        choices=[item.value for item in ResidualStrategy],
        default=[item.value for item in ResidualStrategy],
    )

    cloud = commands.add_parser(
        "cloud-digitalocean",
        help="execute an allowlisted campaign on one fail-safe GPU Droplet",
    )
    cloud.add_argument("plan", type=Path)
    cloud.add_argument("--dry-run", action="store_true")
    return parser


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def command_smoke_model(arguments: argparse.Namespace) -> int:
    device = resolve_device(arguments.device)
    config = ModelConfig(
        vocab_size=128,
        max_sequence_length=32,
        n_layers=arguments.layers,
        d_model=64,
        n_heads=4,
        d_ff=128,
        residual_strategy=ResidualStrategy(arguments.strategy),
    )
    model = ReZeroLM(config).to(device)
    token_ids = torch.randint(3, config.vocab_size, (2, 32), device=device)
    output = model(token_ids, token_ids.roll(-1, dims=1))
    assert output.loss is not None
    output.loss.backward()
    payload = {
        "device": device,
        "strategy": arguments.strategy,
        "parameters": model.parameter_count(),
        "loss": float(output.loss.detach()),
        "gates": [to_primitive(stat) for stat in collect_gate_stats(model)],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_smoke_lab(arguments: argparse.Namespace) -> int:
    device = resolve_device(arguments.device)
    registry = Registry(arguments.root)
    budget = Budget(max_wall_seconds=600, max_device_hours=0.25, max_failures=0)
    hypothesis = Hypothesis(
        claim="The selected residual strategy completes a finite synthetic mini-pilot.",
        baseline="pre_rms",
        candidate=arguments.strategy,
        context={"dataset": "synthetic", "depth": 4},
        independent_variables=("residual_strategy",),
        dependent_variables=("loss", "finite_state"),
        prediction="candidate_completes",
        minimum_effect=arguments.minimum_effect,
        falsification=("Any non-finite model or optimizer state invalidates the run.",),
        budget=budget,
    )
    registry.register_hypothesis(hypothesis)
    model_config = ModelConfig(
        vocab_size=128,
        max_sequence_length=32,
        n_layers=4,
        d_model=64,
        n_heads=4,
        d_ff=128,
        residual_strategy=ResidualStrategy(arguments.strategy),
    )
    training = TrainConfig(steps=arguments.steps, batch_size=2, seed=7, device=device)
    assert hypothesis.hypothesis_id is not None
    manifest = RunManifest(
        hypothesis_id=hypothesis.hypothesis_id,
        stage=RunStage.MINI_PILOT,
        seed=training.seed,
        config=staged_config(model=model_config, training=training),
        budget=budget,
    )
    result = run_registered_synthetic(registry, manifest)
    print(json.dumps(to_primitive(result), indent=2, sort_keys=True))
    return 0 if result.failure is None else 1


def command_prepare_data(arguments: argparse.Namespace) -> int:
    paths = prepare_public_corpus(arguments.name, arguments.root)
    print(json.dumps([str(path) for path in paths], indent=2))
    return 0


def command_run_manifest(arguments: argparse.Namespace) -> int:
    values = json.loads(arguments.manifest.read_text())
    if not isinstance(values, dict):
        raise TypeError("manifest JSON must contain an object")
    manifest = RunManifest.from_dict(values)
    if manifest.config.get("runner") == "governed_control_v1":
        if arguments.data_root is None or arguments.governed_dataset_manifest is None:
            raise ValueError(
                "governed control manifests require --data-root and dataset manifest"
            )
        import celiums_rezero.governed.schemas as governed_schemas
        from celiums_rezero.governed.lab import run_registered_governed

        dataset_values = json.loads(arguments.governed_dataset_manifest.read_text())
        policy = SufficiencyPolicy(**dataset_values["policy"])
        splits = tuple(
            (name, governed_schemas.DatasetSplit(**item))
            for name, item in sorted(dataset_values["splits"].items())
        )
        dataset_manifest = governed_schemas.GovernedDatasetManifest(
            splits=splits,
            policy=policy,
            maximum_evidence_items=dataset_values["maximum_evidence_items"],
            dataset_id=dataset_values["dataset_id"],
        )
        result = run_registered_governed(
            Registry(arguments.registry),
            manifest,
            data_root=arguments.data_root,
            dataset_manifest=dataset_manifest,
        )
        print(json.dumps(to_primitive(result), indent=2, sort_keys=True))
        return 0 if result.failure is None else 1
    result = run_manifest(
        Registry(arguments.registry),
        manifest,
        data_root=arguments.data_root,
    )
    print(json.dumps(to_primitive(result), indent=2, sort_keys=True))
    return 0 if result.failure is None else 1


def command_pilot_wikitext2(arguments: argparse.Namespace) -> int:
    device = resolve_device(arguments.device)
    train_path, validation_path, test_path = wikitext2_paths(arguments.data_root)
    registry = Registry(arguments.run_root)
    required = {arguments.baseline, arguments.candidate}
    if not required <= set(arguments.strategies):
        raise ValueError("campaign strategies must include the baseline and candidate")
    budget = Budget(
        max_wall_seconds=arguments.max_wall_seconds,
        max_device_hours=arguments.max_device_hours,
        max_failures=arguments.max_failures,
        max_artifact_bytes=arguments.max_artifact_bytes,
    )
    hypothesis = Hypothesis(
        claim="Residual strategies differ in finite byte-level WikiText-2 optimization.",
        baseline=arguments.baseline,
        candidate=arguments.candidate,
        context={
            "dataset": "wikitext2",
            "tokenizer": "byte_v1",
            "depth": arguments.layers,
        },
        independent_variables=("residual_strategy",),
        dependent_variables=("validation_nll", "finite_state"),
        prediction=f"{arguments.candidate}_has_lower_validation_nll",
        minimum_effect=arguments.minimum_effect,
        falsification=("Any non-finite model or optimizer state invalidates a run.",),
        budget=budget,
    )
    registry.register_hypothesis(hypothesis)
    assert hypothesis.hypothesis_id is not None

    records = []
    for strategy in arguments.strategies:
        for seed in arguments.seeds:
            model = ModelConfig(
                vocab_size=ByteTokenizer.vocab_size,
                max_sequence_length=arguments.sequence_length,
                n_layers=arguments.layers,
                d_model=arguments.d_model,
                n_heads=4,
                d_ff=arguments.d_model * 2,
                residual_strategy=ResidualStrategy(strategy),
            )
            training = TrainConfig(
                steps=arguments.steps,
                batch_size=arguments.batch_size,
                evaluation_batch_size=arguments.batch_size,
                validation_every_steps=arguments.validation_every_steps,
                validation_nll_threshold=arguments.validation_nll_threshold,
                seed=seed,
                device=device,
            )
            manifest = RunManifest(
                hypothesis_id=hypothesis.hypothesis_id,
                stage=RunStage(arguments.stage),
                seed=training.seed,
                config=staged_corpus_config(
                    model=model,
                    training=training,
                    data_root=arguments.data_root,
                    train_path=train_path,
                    validation_path=validation_path,
                    test_path=test_path,
                    train_limit=arguments.train_bytes,
                    evaluation_limit=arguments.evaluation_bytes,
                ),
                budget=budget,
                data_revision=f"pytorch-examples:{WIKITEXT2_REVISION}",
            )
            records.append(
                (
                    manifest,
                    run_registered_corpus(
                        registry,
                        manifest,
                        data_root=arguments.data_root,
                    ),
                )
            )
    summary = summarize_campaign(
        records,
        baseline=arguments.baseline,
        candidate=arguments.candidate,
        minimum_effect=arguments.minimum_effect,
    )
    report_paths = render_campaign_report(arguments.run_root, summary)
    payload = {
        "campaign": to_primitive(summary),
        "reports": [str(path) for path in report_paths],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(result.failure is None for _, result in records) else 1


def command_pilot_enwiki8(arguments: argparse.Namespace) -> int:
    device = resolve_device(arguments.device)
    path = enwiki8_path(arguments.data_root)
    required = {arguments.baseline, arguments.candidate}
    if not required <= set(arguments.strategies):
        raise ValueError("campaign strategies must include the baseline and candidate")
    registry = Registry(arguments.run_root)
    budget = Budget(
        max_wall_seconds=arguments.max_wall_seconds,
        max_device_hours=arguments.max_device_hours,
        max_failures=arguments.max_failures,
        max_artifact_bytes=arguments.max_artifact_bytes,
    )
    hypothesis = Hypothesis(
        claim="Residual strategies differ on canonical raw-byte enwiki8 optimization.",
        baseline=arguments.baseline,
        candidate=arguments.candidate,
        context={"dataset": "enwiki8", "tokenizer": "raw_byte_v1"},
        independent_variables=("residual_strategy",),
        dependent_variables=("validation_nll", "finite_state"),
        prediction=f"{arguments.candidate}_has_lower_validation_nll",
        minimum_effect=arguments.minimum_effect,
        falsification=("Any non-finite model or optimizer state invalidates a run.",),
        budget=budget,
    )
    registry.register_hypothesis(hypothesis)
    assert hypothesis.hypothesis_id is not None
    records = []
    for strategy in arguments.strategies:
        for seed in arguments.seeds:
            model = ModelConfig(
                vocab_size=256,
                max_sequence_length=arguments.sequence_length,
                n_layers=arguments.layers,
                d_model=arguments.d_model,
                n_heads=8,
                d_ff=arguments.d_model * 2,
                residual_strategy=ResidualStrategy(strategy),
            )
            training = TrainConfig(
                steps=arguments.steps,
                batch_size=arguments.batch_size,
                evaluation_batch_size=arguments.batch_size,
                validation_every_steps=arguments.validation_every_steps,
                validation_nll_threshold=arguments.validation_nll_threshold,
                seed=seed,
                device=device,
            )
            manifest = RunManifest(
                hypothesis_id=hypothesis.hypothesis_id,
                stage=RunStage.PILOT,
                seed=seed,
                config=staged_corpus_config(
                    model=model,
                    training=training,
                    data_root=arguments.data_root,
                    train_path=path,
                    validation_path=path,
                    test_path=path,
                    train_limit=min(arguments.train_bytes, ENWIKI8_SPLITS["train"][1]),
                    evaluation_limit=arguments.evaluation_bytes,
                    train_start=ENWIKI8_SPLITS["train"][0],
                    validation_start=ENWIKI8_SPLITS["validation"][0],
                    test_start=ENWIKI8_SPLITS["test"][0],
                    byte_offset=0,
                ),
                budget=budget,
                data_revision=f"enwiki8:{ENWIKI8_SHA256}:raw-byte-v1",
            )
            records.append(
                (
                    manifest,
                    run_registered_corpus(
                        registry,
                        manifest,
                        data_root=arguments.data_root,
                    ),
                )
            )
    summary = summarize_campaign(
        records,
        baseline=arguments.baseline,
        candidate=arguments.candidate,
        minimum_effect=arguments.minimum_effect,
    )
    report_paths = render_campaign_report(arguments.run_root, summary)
    print(
        json.dumps(
            {"campaign": to_primitive(summary), "reports": [str(p) for p in report_paths]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(result.failure is None for _, result in records) else 1


def command_cloud_digitalocean(arguments: argparse.Namespace) -> int:
    values = json.loads(arguments.plan.read_text())
    if not isinstance(values, dict):
        raise TypeError("cloud plan JSON must contain an object")
    plan = CloudCampaignPlan(
        name=str(values["name"]),
        region=str(values["region"]),
        size=str(values["size"]),
        image=str(values["image"]),
        ssh_key_id=str(values["ssh_key_id"]),
        ssh_private_key=Path(str(values["ssh_private_key"])),
        repository_url=str(values["repository_url"]),
        revision=str(values["revision"]),
        data_command=tuple(str(item) for item in values["data_command"]),
        campaign_command=tuple(str(item) for item in values["campaign_command"]),
        artifact_directory=Path(str(values["artifact_directory"])),
        hourly_rate_usd=float(values["hourly_rate_usd"]),
        max_lifetime_seconds=int(values["max_lifetime_seconds"]),
        max_cost_usd=float(values["max_cost_usd"]),
    )
    summary = execute_digitalocean_campaign(plan, dry_run=arguments.dry_run)
    print(json.dumps(to_primitive(summary), indent=2, sort_keys=True))
    return 0 if summary.status in {"completed", "dry_run"} else 1


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    commands = {
        "smoke-model": command_smoke_model,
        "smoke-lab": command_smoke_lab,
        "prepare-data": command_prepare_data,
        "run-manifest": command_run_manifest,
        "pilot-wikitext2": command_pilot_wikitext2,
        "pilot-enwiki8": command_pilot_enwiki8,
        "cloud-digitalocean": command_cloud_digitalocean,
    }
    return commands[arguments.command](arguments)


if __name__ == "__main__":
    raise SystemExit(main())
