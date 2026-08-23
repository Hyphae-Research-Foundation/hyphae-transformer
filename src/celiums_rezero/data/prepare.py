"""Safe preparation helpers for public historical corpora."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ENWIKI8_URL = "https://mattmahoney.net/dc/enwik8.zip"
ENWIKI8_SHA256 = "547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc"
ENWIKI8_SIZE = 100_000_000
ENWIKI8_SPLITS = {
    "train": (0, 90_000_000),
    "validation": (90_000_000, 5_000_000),
    "test": (95_000_000, 5_000_000),
}
WIKITEXT2_REVISION = "acc295dc7b90714f1bf47f06004fc19a7fe235c4"
WIKITEXT2_BASE_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/"
    f"{WIKITEXT2_REVISION}/word_language_model/data/wikitext-2"
)
WIKITEXT2_SPLITS = {
    "train.txt": "9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f",
    "valid.txt": "f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8",
    "test.txt": "d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"checksum mismatch for {path}: {observed}")


def download_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "hyphae-transformer/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if expected_sha256 is not None:
        try:
            verify_file(temporary, expected_sha256)
        except ValueError:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)
    return destination


def extract_zip(archive: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as compressed:
        for member in compressed.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe archive member: {member.filename}")
            compressed.extract(member, destination)
            if not member.is_dir():
                extracted.append(target)
    return extracted


def prepare_public_corpus(name: str, root: Path) -> list[Path]:
    if name == "wikitext2":
        destination = root / name
        paths: list[Path] = []
        for filename, checksum in WIKITEXT2_SPLITS.items():
            path = destination / filename
            if path.exists():
                verify_file(path, checksum)
            else:
                download_file(
                    f"{WIKITEXT2_BASE_URL}/{filename}",
                    path,
                    expected_sha256=checksum,
                )
            paths.append(path)
        return paths
    if name != "enwiki8":
        raise ValueError(f"unknown corpus: {name}")
    archive = root / "downloads" / f"{name}.zip"
    if archive.exists():
        verify_file(archive, ENWIKI8_SHA256)
    else:
        download_file(ENWIKI8_URL, archive, expected_sha256=ENWIKI8_SHA256)
    extracted = extract_zip(archive, root / name)
    expected = (root / name / "enwik8").resolve()
    if extracted != [expected] or expected.stat().st_size != ENWIKI8_SIZE:
        raise ValueError("enwiki8 archive must contain one 100,000,000-byte enwik8 file")
    return extracted


def wikitext2_paths(root: Path) -> tuple[Path, Path, Path]:
    directory = root / "wikitext2"
    paths = (
        directory / "train.txt",
        directory / "valid.txt",
        directory / "test.txt",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "WikiText-2 is not prepared; run `hyphae-transformer prepare-data wikitext2`. "
            f"Missing: {', '.join(missing)}"
        )
    return paths


def enwiki8_path(root: Path) -> Path:
    path = root / "enwiki8" / "enwik8"
    if not path.is_file() or path.stat().st_size != ENWIKI8_SIZE:
        raise FileNotFoundError(
            "enwiki8 is not prepared; run `hyphae-transformer prepare-data enwiki8`."
        )
    return path
