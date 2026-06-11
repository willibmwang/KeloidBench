#!/usr/bin/env python3
"""Collect local seed files and source manifests for spheroid MVP datasets."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("data/raw/source_manifest.json")


@dataclass(frozen=True)
class LocalSeedFile:
    """A local file that should be copied into the raw data tree."""

    name: str
    source: Path
    destination: Path
    description: str
    required: bool = False


@dataclass(frozen=True)
class RemoteSource:
    """A public source to record and optionally probe for availability."""

    name: str
    url: str
    dataset: str
    description: str


DEFAULT_LOCAL_SEED_FILES = [
    LocalSeedFile(
        name="choi_pdf",
        source=Path("spheroid_2.pdf"),
        destination=Path("data/raw/spheroid/choi/spheroid_2.pdf"),
        description="Choi et al. article PDF.",
    ),
    LocalSeedFile(
        name="choi_supp_pdf",
        source=Path("supplementary_choi.pdf"),
        destination=Path("data/raw/spheroid/choi/supplementary_choi.pdf"),
        description="Choi supplementary PDF.",
    ),
    LocalSeedFile(
        name="choi_supp_xlsx",
        source=Path("supplementary_choi.xlsx"),
        destination=Path("data/raw/spheroid/choi/supplementary_choi.xlsx"),
        description="Choi supplementary Excel data.",
        required=True,
    ),
    LocalSeedFile(
        name="dirand_pdf",
        source=Path("spheroid_1.pdf"),
        destination=Path("data/raw/spheroid/dirand/spheroid_1.pdf"),
        description="Dirand et al. article PDF.",
    ),
]

DEFAULT_REMOTE_SOURCES = [
    RemoteSource(
        name="choi_article",
        url="https://doi.org/10.1038/s42003-024-07194-2",
        dataset="dataset_a",
        description="Keloid fibroblast endothelial spheroid paper.",
    ),
    RemoteSource(
        name="dirand_article",
        url="https://doi.org/10.3390/biomedicines11092350",
        dataset="dataset_a",
        description="Keloid fibroblast spheroid deactivation control paper.",
    ),
    RemoteSource(
        name="bodenmiller_zenodo",
        url="https://doi.org/10.5281/zenodo.4271910",
        dataset="dataset_b",
        description="Public quantified 3D spheroid morphology data.",
    ),
    RemoteSource(
        name="spheroscan_training_zenodo",
        url="https://doi.org/10.5281/zenodo.7555467",
        dataset="dataset_b",
        description="SpheroScan training/supporting data.",
    ),
    RemoteSource(
        name="spheroscan_external_test_zenodo",
        url="https://doi.org/10.5281/zenodo.8211845",
        dataset="dataset_b",
        description="SpheroScan external test data.",
    ),
    RemoteSource(
        name="GSE7980",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7980",
        dataset="dataset_c",
        description="Public keloid gene expression dataset.",
    ),
    RemoteSource(
        name="GSE44270",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44270",
        dataset="dataset_c",
        description="Public keloid gene expression dataset.",
    ),
    RemoteSource(
        name="GSE145725",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145725",
        dataset="dataset_c",
        description="Public keloid gene expression dataset.",
    ),
    RemoteSource(
        name="E-MTAB-2509",
        url="https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-2509",
        dataset="dataset_c",
        description="Public keloid ArrayExpress study.",
    ),
    RemoteSource(
        name="E-MTAB-4945",
        url="https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-4945",
        dataset="dataset_c",
        description="Public keloid ArrayExpress study.",
    ),
]


def resolve_project_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def repo_relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def parse_seed_file(value: str) -> LocalSeedFile:
    """Parse NAME=SOURCE:DESTINATION for ad hoc future seed files."""
    try:
        name, paths = value.split("=", maxsplit=1)
        source, destination = paths.split(":", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected seed file format NAME=SOURCE:DESTINATION"
        ) from exc

    return LocalSeedFile(
        name=name,
        source=Path(source),
        destination=Path(destination),
        description="User-provided local seed file.",
    )


def copy_seed_file(seed_file: LocalSeedFile, root: Path, overwrite: bool) -> dict:
    source = resolve_project_path(root, seed_file.source)
    destination = resolve_project_path(root, seed_file.destination)
    record = {
        **asdict(seed_file),
        "source": repo_relative(root, source),
        "destination": repo_relative(root, destination),
        "exists": source.exists(),
        "copied": False,
    }

    if not source.exists():
        record["error"] = "source file not found"
        return record

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite and source.resolve() != destination.resolve():
        record["skipped"] = "destination exists; use --overwrite to replace"
        return record

    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    record["copied"] = True
    return record


def probe_remote_source(source: RemoteSource, timeout: float, skip_url_check: bool) -> dict:
    record = asdict(source)
    if skip_url_check:
        record["status"] = "not_checked"
        return record

    try:
        response = requests.head(source.url, allow_redirects=True, timeout=timeout)
        record.update(
            {
                "status_code": response.status_code,
                "final_url": response.url,
                "reachable": response.ok,
            }
        )
    except Exception as exc:  # noqa: BLE001
        record.update({"reachable": False, "error": str(exc)})
    return record


def build_manifest(args: argparse.Namespace) -> dict:
    root = args.project_root.resolve()
    local_seed_files = [*DEFAULT_LOCAL_SEED_FILES, *args.seed_file]

    copied_local_files = {}
    local_seed_details = {}
    for seed_file in local_seed_files:
        record = copy_seed_file(seed_file, root, args.overwrite)
        copied_local_files[seed_file.name] = bool(record["copied"])
        local_seed_details[seed_file.name] = record

    remote_details = {
        source.name: probe_remote_source(source, args.timeout, args.skip_url_check)
        for source in DEFAULT_REMOTE_SOURCES
    }

    missing_required = [
        seed_file.name
        for seed_file in local_seed_files
        if seed_file.required and not local_seed_details[seed_file.name]["exists"]
    ]

    return {
        "project_root": str(root),
        "copied_local_files": copied_local_files,
        "local_seed_files": local_seed_details,
        "source_urls": remote_details,
        "missing_required_seed_files": missing_required,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect local seed files and write a source manifest for the spheroid MVP. "
            "Future datasets can be added with repeated --seed-file entries."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--skip-url-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--seed-file",
        type=parse_seed_file,
        action="append",
        default=[],
        help="Extra local seed file as NAME=SOURCE:DESTINATION.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest = build_manifest(args)

    out_path = resolve_project_path(root, args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
