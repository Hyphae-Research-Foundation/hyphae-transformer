#!/usr/bin/env python3
"""Certify Hyphae Transformer against exact tagged Hyphae 2.1.0 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

# The embedded probes run under the independently installed tagged SDK.
# ruff: noqa: E501

HYPHAE_VERSION = "2.1.0"
HYPHAE_TAG_COMMIT = "34b939fc0064b701cc2b34cf6f3a1f07d743638d"
HYPHAE_TAG_TREE = "2f2032f79cb26efa6a928aeafeb52023f5dd572e"
COLLECTION = 13


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--hyphae-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-scale", type=float, default=1.0)
    arguments = parser.parse_args()
    report = certify(**vars(arguments))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def certify(
    *,
    binary: Path,
    wheel: Path,
    python: Path,
    work_root: Path,
    hyphae_source: Path,
    out: Path,
    score_scale: float,
) -> dict[str, object]:
    del out
    source_commit = _run(["git", "rev-parse", "HEAD"], cwd=hyphae_source)
    source_tree = _run(["git", "rev-parse", "HEAD^{tree}"], cwd=hyphae_source)
    if source_commit != HYPHAE_TAG_COMMIT or source_tree != HYPHAE_TAG_TREE:
        raise RuntimeError("Hyphae source is not the exact v2.1.0 tag tree")
    if wheel.name != "hyphae_sdk-2.1.0-py3-none-any.whl":
        raise RuntimeError("Hyphae SDK wheel is not the exact 2.1.0 coordinate")
    work_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    data = work_root / "native"
    endpoint = work_root / "hyphae.sock"
    key_file = work_root / "owner.key"
    receipts = work_root / "receipts"
    _run_json([str(binary), "init", "--data-dir", str(data)])
    collection_digest = _create_collection(python, endpoint, binary, data)
    _run_json(
        [str(binary), "search", "--data-dir", str(data), "provision", "--collection", "13"]
    )
    _run_json(
        [
            str(binary),
            "security",
            "--data-dir",
            str(data),
            "bootstrap",
            "--name",
            "hyphae-transformer-certification-owner",
            "--label",
            "hyphae-transformer-2.1.0",
            "--key-out",
            str(key_file),
        ]
    )
    daemon = _start(binary, data, endpoint, work_root)
    try:
        backend_id, capabilities = _identity(python, endpoint, key_file)
        first = _replay_probe(python, endpoint, key_file, expected_replay=False)
        _stop(daemon, endpoint)
        daemon = _start(binary, data, endpoint, work_root)
        replay_backend, replay_capabilities = _identity(python, endpoint, key_file)
        second = _replay_probe(python, endpoint, key_file, expected_replay=True)
        conformance = json.loads(
            _run(
                [
                    str(python),
                    str(
                        Path(__file__).with_name("hyphae_knowledge_conformance.py")
                    ),
                    "--endpoint",
                    str(endpoint),
                    "--api-key-file",
                    str(key_file),
                    "--collection",
                    str(COLLECTION),
                    "--receipts",
                    str(receipts),
                    "--routing-database",
                    str(work_root / "routing.sqlite3"),
                    "--backend-id",
                    backend_id,
                    "--expected-sdk-version",
                    HYPHAE_VERSION,
                    "--expected-runtime-version",
                    HYPHAE_VERSION,
                    "--score-scale",
                    str(score_scale),
                ]
            )
        )
    finally:
        _stop(daemon, endpoint)
    version = _run_json([str(binary), "version", "--json"])
    scores = first["raw_scores"]
    operationally_compatible = (
        backend_id == replay_backend
        and capabilities == replay_capabilities
        and first["strategy"] == second["strategy"] == "exact_filtered"
        and first["idempotent_replay"] is False
        and second["idempotent_replay"] is True
        and conformance["status"] == "passed"
    )
    observed_maximum = max(scores) if scores else None
    calibrated_maximum = (
        None if observed_maximum is None else min(observed_maximum / score_scale, 1.0)
    )
    requires_recalibration = calibrated_maximum is None or calibrated_maximum < 0.72
    return {
        "schema": "hyphae-transformer.hyphae-native-certification/v1",
        "completed": True,
        "passed": operationally_compatible and not requires_recalibration,
        "operationally_compatible": operationally_compatible,
        "production_compatible": operationally_compatible and not requires_recalibration,
        "hyphae_version": HYPHAE_VERSION,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "binary_sha256": _sha256(binary),
        "wheel_sha256": _sha256(wheel),
        "binary_version": version,
        "capabilities": capabilities,
        "backend_id": backend_id,
        "collection_definition_sha256": collection_digest,
        "first_ingest": first,
        "restart_replay": second,
        "knowledge_conformance": conformance,
        "score_calibration": {
            "raw_scores": scores,
            "observed_maximum": observed_maximum,
            "certified_score_scale": score_scale,
            "calibrated_maximum": calibrated_maximum,
            "current_minimum_score": 0.72,
            "requires_recalibration": requires_recalibration,
        },
    }


def _create_collection(python: Path, endpoint: Path, binary: Path, data: Path) -> str:
    daemon = _start(binary, data, endpoint, data.parent, authenticated=False)
    try:
        script = Path(__file__).with_name("hyphae_210_collection.py")
        if not script.is_file():
            raise RuntimeError("pinned Hyphae collection helper is absent")
        result = json.loads(_run([str(python), str(script), str(endpoint)]))
        return str(result["collection_definition_sha256"])
    finally:
        _stop(daemon, endpoint)


def _identity(python: Path, endpoint: Path, key_file: Path) -> tuple[str, object]:
    code = """  # noqa: E501
import hashlib,json,sys
from pathlib import Path
import hyphae_sdk
from hyphae_sdk.v2 import HyphaeClient
from hyphae_sdk.v2.protocol import PROTOCOL_MAJOR,PROTOCOL_MINOR
key=Path(sys.argv[2]).read_text(encoding='ascii').strip()
with HyphaeClient.local_authenticated(sys.argv[1],key) as client:
    status=client.admin('status')
    capabilities=client.capabilities()
lineage=status.value['snapshot']['directory_lineage']
print(json.dumps({'backend_id':hashlib.sha256(lineage).hexdigest(),'capabilities':capabilities.value,'sdk_version':hyphae_sdk.__version__,'protocol':[PROTOCOL_MAJOR,PROTOCOL_MINOR]},default=lambda x:x.hex() if isinstance(x,bytes) else x))
"""
    value = json.loads(_run([str(python), "-c", code, str(endpoint), str(key_file)]))
    if value["sdk_version"] != HYPHAE_VERSION or value["protocol"] != [1, 5]:
        raise RuntimeError("Hyphae SDK or protocol identity differs")
    return str(value["backend_id"]), value["capabilities"]


def _replay_probe(
    python: Path, endpoint: Path, key_file: Path, *, expected_replay: bool
) -> dict[str, Any]:
    code = """  # noqa: E501
import hashlib,json,sys
from pathlib import Path
from hyphae_sdk.v2 import HyphaeClient,RequestOptions
key=Path(sys.argv[2]).read_text(encoding='ascii').strip(); expected=sys.argv[3]=='1'
body='Exact semantic replay survives daemon restart.'
batch={'idempotency_id':90000000000000000000000000000001,'documents':[{'object_id':90000000000000000000000000000002,'text':body,'doc_values':{'body':body,'source_id':'restart_probe','source_version':'v1','content_digest':hashlib.sha256(body.encode()).hexdigest(),'corpus_generation':'generation-restart-probe-v1','byte_start':0,'byte_end':len(body.encode()),'chunk_ordinal':0},'vectors':{'semantic':[1.0,0.0]}}]}
request={'lexical':{'query':'exact semantic replay','candidate_limit':8,'weight':1},'vectors':[{'target':'semantic','query':[1.0,0.0],'candidate_limit':8,'weight':1,'execution':{'kind':'exact'}}],'filter':{'kind':'compare','field':'corpus_generation','operator':'equal','value':'generation-restart-probe-v1'},'sort':[],'facets':[],'aggregations':[],'limit':8}
with HyphaeClient.local_authenticated(sys.argv[1],key) as client:
    ingested=client.search_ingest(13,batch,options=RequestOptions(durability='strict'))
    searched=client.search_collection(13,request)
branch=searched.value['vector_branches'][0]
if ingested.value['idempotent_replay'] is not expected or branch['strategy']!='exact_filtered' or branch['approximate'] or searched.value['approximate'] or not branch['exact_reranked']:
    raise RuntimeError('replay or exact strategy evidence failed')
print(json.dumps({'idempotent_replay':ingested.value['idempotent_replay'],'commit_csn':ingested.value['commit']['commit_csn'],'strategy':branch['strategy'],'approximate':searched.value['approximate'],'raw_scores':[hit['score'] for hit in searched.value['hits']]}))
"""
    return json.loads(
        _run(
            [
                str(python),
                "-c",
                code,
                str(endpoint),
                str(key_file),
                "1" if expected_replay else "0",
            ]
        )
    )


def _start(
    binary: Path,
    data: Path,
    endpoint: Path,
    root: Path,
    *,
    authenticated: bool = True,
) -> subprocess.Popen[str]:
    command = [str(binary), "serve", "--data-dir", str(data), "--endpoint", str(endpoint)]
    if authenticated:
        command.append("--native-api-key-auth")
    process = subprocess.Popen(
        command,
        stdout=(root / "daemon.stdout.log").open("a"),
        stderr=(root / "daemon.stderr.log").open("a"),
        text=True,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if endpoint.is_socket():
            return process
        if process.poll() is not None:
            raise RuntimeError("Hyphae daemon exited before readiness")
        time.sleep(0.1)
    _stop(process, endpoint)
    raise TimeoutError("Hyphae daemon did not become ready")


def _stop(process: subprocess.Popen[str], endpoint: Path) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    endpoint.unlink(missing_ok=True)


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _run_json(command: list[str]) -> dict[str, object]:
    value = json.loads(_run(command))
    if not isinstance(value, dict):
        raise RuntimeError("Hyphae CLI response was not an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
